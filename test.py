import asyncio
from rich import print
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def simple_test():
    """간단한 테스트"""
    model = ChatOllama(model="qwen3:32b")
    
    servers = {
        "sequential-thinking": {
            "command": "npx", 
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "transport": "stdio"
        }
    }
    
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    
    response = await agent.ainvoke({
        "messages": "간단한 수학 문제를 단계별로 풀어주세요: 2x + 3 = 11"
    })
    
    print("🔍 Sequential Thinking 결과:")
    if hasattr(response, 'messages') and response.messages:
        print(response.messages[-1].content)
    else:
        print(response)

if __name__ == "__main__":
    print("🚀 Sequential Thinking MCP 서버 테스트 시작...")
    
    # 간단한 테스트 실행
    print("\n1️⃣ 간단한 테스트:")
    asyncio.run(simple_test())