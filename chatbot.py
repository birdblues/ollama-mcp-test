# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "textual>=0.40.0",
#     "langchain-ollama>=0.1.0",
#     "langchain-core>=0.2.0",
# ]
# ///

from textual import on, work
from textual.app import App, ComposeResult
from textual.widgets import Input, Footer, RichLog, Static
from textual.events import Key, Paste
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from rich.live import Live
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import requests
import pyperclip

# ChatMessage 클래스 제거 - rich print로 대체

class LangChainOllamaChat(App):
    """LangChain Ollama를 사용한 채팅 앱"""
    
    # 클립보드 액세스 활성화
    ENABLE_COMMAND_PALETTE = False
    
    CSS = """
    #chat-log {
        height: 1fr;
        margin: 0;
        padding: 1;
        scrollbar-size: 0 0;
    }
    
    #streaming-response {
        dock: bottom;
        height: auto;
        margin: 0 1;
        padding: 1;
        border: round yellow;
    }
    
    #streaming-response.hidden {
        display: none;
    }
    
    #chat-input {
        dock: bottom;
        margin: 0;
        border: round $primary;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.ollama_url = "http://localhost:11434"
        self.model_name = "qwen3:30b-32k"  # 기본 모델
        self.chat_model = None
        self.messages = []  # LangChain 메시지 히스토리
        self.available_models = []
        self.console = Console()  # Rich 콘솔 인스턴스
        self.current_ai_widget = None  # 현재 AI 응답 위젯 추적
        
    def compose(self) -> ComposeResult:
        """UI 구성 - RichLog와 입력창"""
        yield RichLog(id="chat-log", highlight=True, markup=True)
        yield Static(id="streaming-response", classes="hidden")  # 스트리밍 응답용 위젯
        yield Input(value="> ", placeholder="메시지를 입력하세요... (종료하려면 'quit' 입력)", id="chat-input")
        # yield Footer()

    def on_mount(self) -> None:
        """앱 시작시 실행"""
        # 시작 메시지를 Rich로 출력
        self.print_welcome_message()
        
        if self.check_ollama_connection():
            self.initialize_chat_model()
            self.load_available_models()
        
    def print_welcome_message(self):
        """환영 메시지를 RichLog에 출력"""
        chat_log = self.query_one("#chat-log")
        welcome_panel = Panel(
            RichMarkdown("""# 🦙 LangChain Ollama 채팅봇

메시지를 입력해서 대화를 시작하세요!

## 명령어:
- `/models` - 사용 가능한 모델 보기
- `/model <이름>` - 모델 변경
- `/clear` - 대화 내역 초기화
- `/system <메시지>` - 시스템 프롬프트 설정
- `quit` - 종료"""),
            title="환영합니다",
            border_style="blue"
        )
        chat_log.write(welcome_panel)

    def check_ollama_connection(self) -> bool:
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                self.print_system_message("✅ Ollama 서버 연결됨")
                return True
            else:
                self.print_system_message("❌ Ollama 서버 응답 오류")
                return False
        except requests.exceptions.RequestException:
            self.print_system_message("❌ Ollama 서버에 연결할 수 없습니다.\n\n**해결 방법:**\n1. `ollama serve` 명령어로 서버 시작\n2. `ollama pull qwen3:32b` 명령어로 모델 다운로드")
            return False
    
    def initialize_chat_model(self):
        """ChatOllama 모델 초기화"""
        try:
            self.chat_model = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_url,
                temperature=0.7,
                top_p=0.9,
                streaming=True,  # 스트리밍 활성화
                # num_predict=500,  # max_tokens 대신 사용
            )
            # 기본 시스템 메시지 설정
            self.messages = [
                SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친근하고 자세하게 답변해주세요.")
            ]
            self.print_system_message(f"🤖 ChatOllama 모델 초기화 완료: **{self.model_name}**")
        except Exception as e:
            self.print_system_message(f"❌ 모델 초기화 실패: {str(e)}")
    
    def load_available_models(self):
        """사용 가능한 모델 목록 로드"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]
                if not self.available_models:
                    self.print_system_message("⚠️ 설치된 모델이 없습니다.\n\n`ollama pull qwen3:32b`를 실행해주세요.")
            else:
                self.print_system_message("⚠️ 모델 목록을 불러올 수 없습니다.")
        except Exception as e:
            self.print_system_message(f"⚠️ 모델 목록 로드 오류: {str(e)}")

    def print_system_message(self, message: str):
        """시스템 메시지를 RichLog에 출력"""
        chat_log = self.query_one("#chat-log")
        panel = Panel(
            RichMarkdown(message),
            title="시스템",
            border_style="yellow"
        )
        chat_log.write(panel)

    def print_user_message(self, message: str):
        """사용자 메시지를 RichLog에 출력"""
        chat_log = self.query_one("#chat-log")
        panel = Panel(
            RichMarkdown(message),
            title="You",
            border_style="blue"
        )
        chat_log.write(panel)

    def print_ai_message(self, message: str):
        """AI 메시지를 RichLog에 출력"""
        chat_log = self.query_one("#chat-log")
        panel = Panel(
            RichMarkdown(message),
            title="AI",
            # border_style="green"
        )
        chat_log.write(panel)
    
    def update_ai_response(self, partial_message: str):
        """실시간으로 AI 응답을 업데이트"""
        streaming_widget = self.query_one("#streaming-response")
        
        # 스트리밍 상태가 아니라면 위젯을 표시하고 시작
        if not hasattr(self, '_streaming_active') or not self._streaming_active:
            self._streaming_active = True
            streaming_widget.remove_class("hidden")
        
        # 스트리밍 위젯 내용 업데이트
        # panel = Panel(
        #     RichMarkdown(partial_message + " ▌"),
        #     title="AI (작성 중...)",
        #     border_style="yellow"
        # )
        streaming_widget.update(RichMarkdown(partial_message + " ▌"))
    
    def finalize_ai_response(self, final_message: str):
        """스트리밍 완료 후 최종 AI 응답 출력"""
        chat_log = self.query_one("#chat-log")
        streaming_widget = self.query_one("#streaming-response")
        
        # 최종 응답을 채팅 로그에 추가
        # panel = Panel(
        #     RichMarkdown(final_message),
        #     title="AI",
        #     # border_style="green"
        # )
        chat_log.write(RichMarkdown(final_message))
        
        # 스트리밍 위젯 숨기기 및 상태 종료
        streaming_widget.add_class("hidden")
        streaming_widget.update("")  # 내용 초기화
        self._streaming_active = False

    def on_key(self, event: Key) -> None:
        """키 입력 처리"""
        # 포커스된 위젯이 입력창인지 확인
        focused = self.focused
        if focused and focused.id == "chat-input":
            input_widget = focused
            current_value = input_widget.value
            
            # 백스페이스 키이고 현재 값이 "> " 이하인 경우 이벤트 차단
            if event.key == "backspace" and len(current_value) <= 2:
                if current_value.startswith(">"):
                    event.prevent_default()
                    input_widget.value = "> "
                    # 커서를 끝으로 이동
                    input_widget.cursor_position = len(input_widget.value)
            
            # 붙여넣기 단축키 처리 (Cmd+V 또는 Ctrl+V)
            elif event.key == "ctrl+v" or event.key == "cmd+v":
                # Textual의 기본 붙여넣기 동작을 허용
                pass
            
            # 복사 단축키 처리 (Cmd+C 또는 Ctrl+C)
            elif event.key == "ctrl+c" or event.key == "cmd+c":
                # 입력창에서의 복사는 Textual의 기본 동작 허용
                pass
        
        # 전역 복사 단축키 처리 (화면의 모든 선택된 텍스트)
        elif event.key == "ctrl+c" or event.key == "cmd+c":
            try:
                # 현재 선택된 텍스트가 있는지 확인
                selected_text = self.get_selected_text()
                if selected_text:
                    pyperclip.copy(selected_text)
                    event.prevent_default()
            except:
                # pyperclip이 없거나 오류 발생 시 기본 동작 허용
                pass

    def get_selected_text(self) -> str:
        """현재 선택된 텍스트를 가져오기"""
        try:
            # Textual의 현재 선택 영역에서 텍스트 추출
            # 이는 Textual의 내부 구조에 따라 달라질 수 있음
            if hasattr(self, '_selected_text'):
                return self._selected_text
            return ""
        except:
            return ""

    def on_paste(self, _event: Paste) -> None:
        """붙여넣기 이벤트 처리"""
        # 포커스된 위젯이 입력창인지 확인
        focused = self.focused
        if focused and focused.id == "chat-input":
            input_widget = focused
            
            # 현재 커서 위치 확인
            cursor_pos = input_widget.cursor_position
            
            # 프롬프트 부분("> ")을 보호하면서 붙여넣기
            if cursor_pos < 2:
                # 프롬프트 앞에 붙여넣기 시도 시 프롬프트 뒤로 이동
                cursor_pos = 2
                input_widget.cursor_position = cursor_pos
            
            # 기본 붙여넣기 동작 허용
            # Textual이 자동으로 처리하도록 함


    @on(Input.Submitted)
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """사용자 입력 처리"""
        raw_input = event.value.strip()
        
        # '> ' 프롬프트 제거
        if raw_input.startswith("> "):
            user_input = raw_input[2:].strip()
        else:
            user_input = raw_input
        
        if not user_input:
            # 입력이 없으면 프롬프트만 다시 설정
            event.input.value = "> "
            return
            
        # 입력창을 프롬프트로 초기화
        event.input.value = "> "
        
        # 종료 명령어 처리
        if user_input.lower() in ['quit', 'exit', '종료']:
            self.exit()
            return
        
        # 명령어 처리
        if user_input.startswith('/'):
            await self.handle_command(user_input)
            return
            
        # 모델이 초기화되지 않았다면 처리 중단
        if not self.chat_model:
            await self.show_error_message("❌ 모델이 초기화되지 않았습니다. Ollama 서버를 확인해주세요.")
            return
            
        # 사용자 메시지를 Rich로 출력
        self.print_user_message(user_input)
        
        # 메시지 히스토리에 추가
        self.messages.append(HumanMessage(content=user_input))
        
        # AI 응답 생성 시작
        self.generate_response()

    async def handle_command(self, command: str):
        """명령어 처리"""
        
        if command == "/models":
            if self.available_models:
                models_text = "**사용 가능한 모델들:**\n\n" + "\n".join(f"• {model}" for model in self.available_models)
                models_text += f"\n\n**현재 모델:** {self.model_name}"
                self.print_system_message(models_text)
            else:
                self.print_system_message("❌ 사용 가능한 모델이 없습니다.\n\n`ollama pull <모델명>`으로 모델을 다운로드하세요.")
                
        elif command.startswith("/model "):
            new_model = command.split(" ", 1)[1].strip()
            if new_model in self.available_models:
                self.model_name = new_model
                self.initialize_chat_model()  # 새 모델로 재초기화
                self.print_system_message(f"✅ 모델이 **{new_model}**로 변경되었습니다.")
            else:
                self.print_system_message(f"❌ 모델 '{new_model}'를 찾을 수 없습니다.\n\n`/models`로 사용 가능한 모델을 확인하세요.")
                
        elif command.startswith("/system "):
            system_prompt = command.split(" ", 1)[1].strip()
            # 기존 시스템 메시지 교체
            self.messages = [SystemMessage(content=system_prompt)] + [msg for msg in self.messages if not isinstance(msg, SystemMessage)]
            self.print_system_message(f"✅ 시스템 프롬프트가 설정되었습니다:\n\n*{system_prompt}*")
                
        elif command == "/clear":
            # 시스템 메시지만 유지하고 나머지 초기화
            system_messages = [msg for msg in self.messages if isinstance(msg, SystemMessage)]
            self.messages = system_messages if system_messages else [
                SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친근하고 자세하게 답변해주세요.")
            ]
            self.print_system_message("✅ 대화 내역이 초기화되었습니다.")
            
        else:
            self.print_system_message(f"❌ 알 수 없는 명령어입니다: {command}\n\n사용 가능한 명령어를 보려면 처음 메시지를 참고하세요.")

    async def show_error_message(self, message: str):
        """오류 메시지 표시"""
        self.print_system_message(message)

    @work(thread=True)
    def generate_response(self) -> None:
        """AI 응답 생성 (LangChain 스트리밍 사용)"""
        try:
            # 최근 10개 메시지만 유지 (메모리 절약)
            if len(self.messages) > 11:  # 시스템 메시지 + 10개
                system_msgs = [msg for msg in self.messages if isinstance(msg, SystemMessage)]
                recent_msgs = self.messages[-10:]
                self.messages = system_msgs + recent_msgs
            
            response_content = ""
            
            # 스트리밍 상태 초기화
            self._streaming_active = False
            
            # LangChain 스트리밍 호출 사용
            for chunk in self.chat_model.stream(self.messages):
                if hasattr(chunk, 'content') and chunk.content:
                    response_content += chunk.content
                    # 실시간으로 AI 응답 업데이트
                    self.call_from_thread(self.update_ai_response, response_content)
            
            # 스트리밍 완료 후 최종 메시지 출력
            if response_content:
                self.messages.append(AIMessage(content=response_content))
                # 스트리밍 완료 표시
                self._streaming_active = False
                self.call_from_thread(self.finalize_ai_response, response_content)
            
        except Exception as e:
            error_msg = f"❌ 오류가 발생했습니다: {str(e)}\n\n모델이 다운로드되어 있는지 확인해주세요."
            self.call_from_thread(self.print_system_message, error_msg)

def main():
    """메인 함수"""
    print("🦙 LangChain Ollama 채팅 앱을 시작합니다...")
    print("\n📋 사전 준비사항:")
    print("1. ollama serve")
    print("2. ollama pull qwen3:32b")
    print("3. python app.py\n")
    
    app = LangChainOllamaChat()
    app.run()

if __name__ == "__main__":
    main()
