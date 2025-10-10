
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv
import os
from agent_framework import WorkflowViz

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
deployment_name = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]

chat_client = AzureOpenAIChatClient(
    deployment_name=deployment_name,
    endpoint=endpoint,
    api_key=api_key,
)

import asyncio
from dataclasses import dataclass
from typing import cast

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    RequestInfoEvent,
    RequestInfoExecutor,
    RequestInfoMessage,
    RequestResponse,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    WorkflowRunState,
    WorkflowStatusEvent,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

"""
Sample: Writer-Reviewer Workflow with Human-in-the-Loop Approval

このサンプルは、WriterとReviewerの2つのエージェントを使用したワークフローに、
人間による承認ステップを追加する方法を示します。

ワークフローの流れ:
1. Writerエージェントがコンテンツを生成
2. Reviewerエージェントが内容をレビューしてフィードバックを提供
3. 人間がレビュー結果を確認し、承認または修正指示を提供
4. 承認された場合は完了、修正指示があればWriterに戻って再作成

主要な概念:
- RequestInfoExecutorを使用した人間との対話
- RequestResponseによる相関されたリクエスト/レスポンスの処理
- ワークフロー内でのループと条件分岐
- 複数のエージェントとカスタムエグゼキューターの連携

前提条件:
- Azure OpenAIの設定と必要な環境変数の構成
- azure-identityによる認証。実行前に `az login` を実行
- WorkflowBuilder、エグゼキューター、エッジ、イベント、ストリーミング実行の基本知識
"""


@dataclass
class HumanReviewRequest(RequestInfoMessage):
    """人間のレビューアーに送信されるリクエストメッセージ。
    
    RequestInfoMessageをサブクラス化することで、リクエストの正確なスキーマを定義し、
    強い型付け、将来互換性のある検証、明確な相関セマンティクスを提供します。
    """

    prompt: str = ""  # この行を追加
    draft_content: str = ""
    reviewer_feedback: str = ""
    iteration: int = 1


class ReviewCoordinator(Executor):
    """レビューフローを調整し、人間の承認を管理するエグゼキューター。
    
    責務:
    - Reviewerからのフィードバックを受け取る
    - 人間のレビューアーにHumanReviewRequestを送信
    - 人間の決定に基づいて、完了またはWriterへの再作成指示を行う
    """

    def __init__(self, writer_id: str, request_info_id: str, coordinator_id: str = "review_coordinator"):
        super().__init__(id=coordinator_id)
        self._writer_id = writer_id
        self._request_info_id = request_info_id

    @handler
    async def handle_reviewer_response(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[HumanReviewRequest],
    ) -> None:
        """Reviewerのフィードバックを処理し、人間の承認を要求する。"""
        
        # 現在の状態を取得して反復回数を追跡
        state = await ctx.get_state() or {}
        draft_content = cast(str, state.get("current_draft", ""))
        iteration = int(state.get("iteration", 0)) + 1
        
        reviewer_feedback = response.agent_run_response.text or ""
        
        print(f"\n{'='*60}")
        print(f"反復 {iteration}: Reviewerのフィードバック")
        print(f"{'='*60}")
        print(reviewer_feedback)
        
        # 状態を更新
        await ctx.set_state({
            "iteration": iteration,
            "current_draft": draft_content,
            "reviewer_feedback": reviewer_feedback,
        })
        
        # 人間のレビューアーにリクエストを送信
        await ctx.send_message(
            HumanReviewRequest(
                prompt=(
                    "レビュー結果を確認してください。\n"
                    "'approve' で承認、または修正指示を入力してください。"
                ),
                draft_content=draft_content,
                reviewer_feedback=reviewer_feedback,
                iteration=iteration,
            ),
            target_id=self._request_info_id,
        )

    @handler
    async def handle_human_decision(
        self,
        feedback: RequestResponse[HumanReviewRequest, str],
        ctx: WorkflowContext[AgentExecutorRequest, str],
    ) -> None:
        """人間の決定を処理し、承認または修正を実行する。"""
        
        human_reply = (feedback.data or "").strip().lower()
        state = await ctx.get_state() or {}
        draft_content = cast(str, state.get("current_draft", ""))
        
        print(f"\n{'='*60}")
        print(f"人間の決定: {feedback.data}")
        print(f"{'='*60}")
        
        if human_reply == "approve":
            # 承認された場合、最終出力として提出
            print("\n✅ コンテンツが承認されました！")
            await ctx.yield_output(draft_content)
            return
        
        # 修正指示がある場合、Writerに戻す
        print(f"\n🔄 修正指示あり。Writerに再作成を依頼します...")
        
        revision_prompt = (
            f"以下のフィードバックに基づいてコンテンツを修正してください:\n\n"
            f"前回の下書き:\n{draft_content}\n\n"
            f"Reviewerのフィードバック:\n{state.get('reviewer_feedback', '')}\n\n"
            f"人間からの修正指示:\n{feedback.data}"
        )
        
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[ChatMessage(Role.USER, text=revision_prompt)],
                should_respond=True,
            ),
            target_id=self._writer_id,
        )


class DraftCapture(Executor):
    """Writerからの下書きをキャプチャし、Reviewerに転送するエグゼキューター。"""

    def __init__(self, reviewer_id: str, capture_id: str = "draft_capture"):
        super().__init__(id=capture_id)
        self._reviewer_id = reviewer_id

    @handler
    async def capture_draft(
        self,
        response: AgentExecutorResponse,
        ctx: WorkflowContext[AgentExecutorRequest],
    ) -> None:
        """Writerの下書きを保存し、Reviewerに送信する。"""
        
        draft_content = response.agent_run_response.text or ""
        
        print(f"\n{'='*60}")
        print("Writerの下書き")
        print(f"{'='*60}")
        print(draft_content)
        
        # 下書きを状態に保存
        state = await ctx.get_state() or {}
        await ctx.set_state({
            **state,
            "current_draft": draft_content,
        })
        
        # Reviewerにレビュー依頼を送信
        review_request = (
            f"以下のコンテンツをレビューし、品質、明確さ、正確性について"
            f"簡潔なフィードバックを提供してください:\n\n{draft_content}"
        )
        
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[ChatMessage(Role.USER, text=review_request)],
                should_respond=True,
            ),
            target_id=self._reviewer_id,
        )


def visualize_workflow(workflow, filename="workflow_diagram"):
    # WorkflowVizオブジェクトを作成
    viz = WorkflowViz(workflow)
    
    # SVGファイルとして保存
    try:
        svg_path = viz.export(format="svg", filename=filename)
        print(f"✅ ワークフロー図が '{svg_path}' に保存されました")
        return svg_path
        
    except ImportError as e:
        print("❌ エラー: 'graphviz'パッケージがインストールされていません")
        print("インストール方法: pip install agent-framework[viz] --pre")
        print(f"詳細: {e}")
        return None
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return None
    
async def main() -> None:
    """Writer-Reviewerワークフローを構築し、human-in-the-loopで実行する。"""
    
    print("Writer-Reviewer Human-in-the-Loop ワークフローを開始します")
    print("="*60)
    
    # Azure OpenAI クライアントを作成
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    
    # Writerエージェントを作成
    writer_agent = chat_client.create_agent(
        name="Writer",
        instructions=(
            "あなたは優秀なコンテンツライターです。"
            "明確で魅力的なコンテンツを作成し、フィードバックに基づいて改善してください。"
        ),
    )
    
    # Reviewerエージェントを作成
    reviewer_agent = chat_client.create_agent(
        name="Reviewer",
        instructions=(
            "あなたは経験豊富なコンテンツレビューアーです。"
            "以下の観点から評価し、実行可能なフィードバックを提供してください:\n"
            "1. 明確さ - 理解しやすいか?\n"
            "2. 完全性 - トピックを十分にカバーしているか?\n"
            "3. 正確性 - 情報は正しいか?\n"
            "フィードバックは簡潔にしてください。"
        ),
    )
    
    # エグゼキューターを作成
    writer = AgentExecutor(writer_agent, id="writer")
    reviewer = AgentExecutor(reviewer_agent, id="reviewer")
    draft_capture = DraftCapture(reviewer_id=reviewer.id)
    request_info = RequestInfoExecutor(id="request_info")
    coordinator = ReviewCoordinator(
        writer_id=writer.id,
        request_info_id=request_info.id,
    )
    
    # ワークフローを構築
    # Writer → DraftCapture → Reviewer → Coordinator → RequestInfo
    #    ↑                                      ↓
    #    └──────────── (修正が必要な場合) ──────┘
    workflow = (
        WorkflowBuilder()
        .set_start_executor(writer)
        .add_edge(writer, draft_capture)
        .add_edge(draft_capture, reviewer)
        .add_edge(reviewer, coordinator)
        .add_edge(coordinator, request_info)
        .add_edge(request_info, coordinator)
        .add_edge(coordinator, writer)
        .build()
    )
    
    visualize_workflow(workflow, "HumanInTheLoop_Workflow")
    # ワークフローを実行
    initial_task = "手頃な価格で楽しい新型電動SUVのスローガンを作成してください。"
    
    pending_responses: dict[str, str] | None = None
    completed = False
    final_output: str | None = None
    
    print(f"\n初期タスク: {initial_task}")
    
    while not completed:
        # 最初の反復ではrun_streamを使用、以降はsend_responses_streamingを使用
        stream = (
            workflow.send_responses_streaming(pending_responses)
            if pending_responses
            else workflow.run_stream(initial_task)
        )
        
        pending_requests: list[tuple[str, HumanReviewRequest]] = []
        
        async for event in stream:
            if isinstance(event, RequestInfoEvent):
                # 人間の入力が必要
                if isinstance(event.data, HumanReviewRequest):
                    pending_requests.append((event.request_id, event.data))
            
            elif isinstance(event, WorkflowOutputEvent):
                # ワークフローが完了
                final_output = cast(str, event.data)
                completed = True
            
            elif isinstance(event, WorkflowStatusEvent):
                if event.state in (
                    WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
                    WorkflowRunState.IN_PROGRESS_PENDING_REQUESTS,
                ):
                    # 人間の入力を待機中
                    pass
        
        # 保留中のリクエストがある場合、人間の入力を収集
        if pending_requests and not completed:
            pending_responses = {}
            
            for request_id, request_data in pending_requests:
                print(f"\n{'='*60}")
                print(f"人間のレビューが必要 (反復 {request_data.iteration})")
                print(f"{'='*60}")
                print(f"\n📝 下書き:\n{request_data.draft_content}")
                print(f"\n💬 Reviewerのフィードバック:\n{request_data.reviewer_feedback}")
                print(f"\n{request_data.prompt}")
                print("\n入力してください ('approve' で承認、または修正指示): ", end="", flush=True)
                
                user_input = input().strip()
                pending_responses[request_id] = user_input
        else:
            # 保留中のリクエストがない場合は完了
            if not completed:
                completed = True
            pending_responses = None
    
    # 最終結果を表示
    if final_output:
        print(f"\n{'='*60}")
        print("✨ 最終承認コンテンツ")
        print(f"{'='*60}")
        print(final_output)
        print(f"\n{'='*60}")
        print("ワークフロー完了!")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())