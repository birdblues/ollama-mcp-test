import asyncio
from rich import print
from rich.console import Console
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

async def simple_test():
    """간단한 테스트"""
    model = ChatOllama(model="qwen3:30b-32k")
    
    servers = {
        "sequential-thinking": {
            "command": "npx", 
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "transport": "stdio",
            "env": {
                "DISABLE_THOUGHT_LOGGING": "true"
            }
        }
    }
    
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    
    console = Console()
    with console.status("[green]수학 문제 풀이 중...\n", spinner="dots"):
        response = await agent.ainvoke({
            "messages": "간단한 수학 문제를 단계별로 풀어주세요: 2x + 3 = 11"
        })
    
    print("🔍 Sequential Thinking 결과:")
    print(response)

if __name__ == "__main__":
    # 간단한 테스트 실행
    asyncio.run(simple_test())
