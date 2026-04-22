import os
import json
import logging
import uuid
import httpx
import datetime
from typing import Annotated, Any, Sequence, Optional, List, Dict
from pydantic import BaseModel, Field
import ast

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from ddgs import DDGS
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing_extensions import TypedDict

def reduce_metadata(left: dict, right: dict) -> dict:
    if not left:
        left = {"token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "trace": []}
    if not right:
        return left
    
    import copy
    new_left = copy.deepcopy(left)
    for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        new_left["token_usage"][k] = new_left["token_usage"].get(k, 0) + right.get("token_usage", {}).get(k, 0)
    
    if "trace" in right:
        new_left["trace"].extend(right["trace"])
        
    return new_left

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    metadata: Annotated[dict, reduce_metadata]

wikipedia = WikipediaAPIWrapper()

@tool("duckduckgo_search")
def web_search(query: str) -> str:
    """Use DuckDuckGo to search the web for recent news, current trending topics, and active internet drama."""
    try:
        results = DDGS().text(query, max_results=3)
        return str(results)
    except Exception as e:
        return f"Search failed: {e}"

@tool("wikipedia_search")
def wiki_search(query: str) -> str:
    """Use Wikipedia to search for historical events, biographies, past incidents, and deep factual lore."""
    return wikipedia.run(query)

@tool("nano_banana_pro_tool")
def nano_banana_pro_tool(prompt: str, image_url: Optional[str] = None) -> str:
    """
    Visual Tool powered by Gemini 3 Pro Image Preview for text-to-image and image editing.
    Generates high-quality visual stories, memes, or graphics.
    Returns the absolute path to the generated image saved locally.
    """
    api_key = os.environ.get("NANO_BANANA_API_KEY")
    if not api_key:
        return "Error: NANO_BANANA_API_KEY not found."
    
    try:
        client = genai.Client(api_key=api_key)
        
        contents = [prompt]
        
        if image_url:
            try:
                import base64
                resp = httpx.get(image_url, timeout=15)
                resp.raise_for_status()
                img_part = types.Part.from_bytes(data=resp.content, mime_type="image/png")
                contents = [img_part, prompt]
            except Exception as img_err:
                contents = [f"{prompt} (Note: Could not fetch reference image: {img_err})"]
        
        result = client.models.generate_content(
            model='gemini-3-pro-image-preview',
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            )
        )
        
        for part in result.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                img_bytes = part.inline_data.data
                ext = part.inline_data.mime_type.split("/")[-1]
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                base_dir = os.getcwd()
                meme_dir = os.path.join(base_dir, "memes", date_str)
                os.makedirs(meme_dir, exist_ok=True)
                filename = f"generated_meme_{uuid.uuid4().hex[:8]}.{ext}"
                filepath = os.path.join(meme_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                
                return f"[GENERATED IMAGE] /memes/{date_str}/{filename} | Prompt: '{prompt}'"
        
        text_response = result.text if hasattr(result, 'text') else "No image generated."
        return f"[NO IMAGE RETURNED] Model responded with text: {text_response}"
    except Exception as e:
        return f"Image generation failed: {e}"

@tool("python_repl")
def python_repl_tool(code: str) -> str:
    """
    A Python shell for the Layout & Polish agent to calculate aspect ratios,
    text formatting positions, chart labels, or perform math.
    """
    try:
        tree = ast.parse(code, mode='exec')
        local_env = {}
        exec(compile(tree, filename="<ast>", mode="exec"), {}, local_env)
        res = []
        for k, v in local_env.items():
            if not k.startswith("__") and not callable(v):
                res.append(f"{k} = {v}")
        if not res:
            return "Execution successful, but no variables saved. Use simple variable assignments."
        return "\n".join(res)
    except Exception as e:
        return f"Python Error: {e}"

def create_agent(llm, tools: list, system_prompt: str, agent_name: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
        
    def agent_node(state: AgentState):
        logger.info(f"--- AGENT {agent_name} RUNNING ---")
        chain = prompt | llm_with_tools
        response = chain.invoke(state)
        
        has_tool_calls = bool(response.tool_calls)
        content_preview = str(response.content)[:200] if response.content else "(empty)"
        logger.info(f"--- AGENT {agent_name} RESULT --- tool_calls={has_tool_calls}, content_preview={content_preview}")
        
        usage = response.response_metadata.get("token_usage", {})
        trace_entry = {
            "agent": agent_name,
            "called_at": "timestamp", 
            "content": response.content,
            "tool_calls": response.tool_calls
        }
        
        response.name = agent_name
        
        return {
            "messages": [response],
            "metadata": {
                "token_usage": {
                     "prompt_tokens": usage.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0),
                     "total_tokens": usage.get("total_tokens", 0),
                },
                "trace": [trace_entry]
            }
        }
        
    return agent_node


class SupervisorOutput(BaseModel):
    reasoning: str = Field(description="Short routing rationale.")
    next: str = Field(description="The next agent to route to, or 'FINISH' if the user's request has been fully addressed.")

def create_supervisor():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    system_prompt = (
        "You are a veteran Chief Meme Officer and Supervisor coordinating a Multi-Agent system for a viral 'Gen Z / Brainrot Meme Factory'.\n"
        "Your workers are:\n"
        "- 'Researcher': A deeply knowledgeable analyst. Uses DuckDuckGo for the latest trends/news/drama, and uses Wikipedia for historical events, past incidents, and deep factual lore required for hyper-niche memes.\n"
        "- 'Visualizer': A veteran content/meme creator aiming for millions of impressions on Twitter. Uses the nano_banana_pro_tool to generate incredibly relatable, high-quality, and scalable viral memes (never using emojis in image text).\n"
        "- 'Layout_Calculator': A Python developer that evaluates aspect ratios, chart math, and layout overlays for maximum cross-platform engagement.\n\n"
        "Goal: Fulfill the user's request by calling the relevant workers in logical sequence to craft an unstoppable viral hit.\n"
        "If a user provides an uploaded image, route to Visualizer or Layout_Calculator if image modifications/processing are needed.\n"
        "If the user asks for a very specific technical diagram, chart, or infographic and you are not 100% confident in the numbers, YOU MUST route to the Researcher FIRST.\n"
        "If the user requests text in multiple languages, or complex text placement (e.g., 'perfect chart labels'), ALWAYS route to the 'Layout_Calculator' before the 'Visualizer' to precisely measure placement logic.\n"
        "Never 'assume' facts if the user is asking for instructional materials. Delegate to Researcher whenever an ambiguous noun or technology is encountered.\n"
        "Once the final meme, story, or visual output generation is confirmed completely, respond with 'FINISH'.\n"
        "Do NOT call a worker multiple times for the exact same repeating task if they have already succeeded.\n"
        "Provide a short rationale and then decide the next route."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Select one of: ['Researcher', 'Visualizer', 'Layout_Calculator', 'FINISH'].")
    ])
    
    supervisor_chain = prompt | llm.with_structured_output(SupervisorOutput, include_raw=True)
    
    def supervisor_node(state: AgentState):
        decision_raw = supervisor_chain.invoke(state)
        decision = decision_raw["parsed"] if isinstance(decision_raw, dict) else None
        
        if decision:
            decision_next = getattr(decision, "next", "FINISH")
            decision_reasoning = getattr(decision, "reasoning", "No reasoning provided.")
        else:
            decision_next = "FINISH"
            decision_reasoning = "Fallback: execution ended prematurely due to parsing error."
            
        raw_message = decision_raw.get("raw") if isinstance(decision_raw, dict) else None
        usage = raw_message.response_metadata.get("token_usage", {}) if raw_message else {}
        
        trace_entry = {
            "agent": "Supervisor",
            "decision": decision_next,
            "reasoning": decision_reasoning
        }
        return {
            "next": decision_next,
            "metadata": {
                "trace": [trace_entry],
                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            }
        }
        
    return supervisor_node


from langgraph.prebuilt import ToolNode

def create_tool_node(tools: list, name: str):
    executor = ToolNode(tools)
    def trace_tool_node(state: AgentState):
        result = executor.invoke(state)
        traces = []
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                traces.append({
                    "agent": "ToolExecution",
                    "tool": msg.name,
                    "output": msg.content
                })
        return {"messages": result["messages"], "metadata": {"trace": traces}}
    return trace_tool_node

def get_app():
    research_tools = [web_search, wiki_search]
    visualizer_tools = [nano_banana_pro_tool]
    layout_tools = [python_repl_tool]
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")
    text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    
    supervisor_node = create_supervisor()
    
    research_prompt = (
        "You are an expert internet Researcher and cultural analyst for meme writing. "
        "Your job is to gather facts that can become high-comedy setups, contrasts, and punchline fuel.\n\n"
        "TOOL ROUTING:\n"
        "1) If the subject is RECENT news, active discourse, sports, drama, or current social trends, use 'duckduckgo_search'.\n"
        "2) If the subject is HISTORICAL events, biographies, timelines, wars, inventions, or cultural lore, use 'wikipedia_search'.\n\n"
        "OUTPUT STYLE:\n"
        "Return concise bullets with only meme-useful facts:\n"
        "- weird contrast facts\n"
        "- number/stat hooks\n"
        "- relatable modern parallels\n"
        "- 3 short joke angles (setup -> twist)"
    )
    research_agent = create_agent(text_llm, research_tools, research_prompt, "Researcher")

    visualizer_prompt = (
        "You are a veteran meme creative director and punchline specialist.\n"
        "Your objective: generate truly funny, high-share memes, not descriptive statements.\n"
        "You MUST call nano_banana_pro_tool for image generation.\n\n"
        "HUMOR MODE SELECTION:\n"
        "- SAFE: broadly relatable, no niche references needed.\n"
        "- SPICY: sharper sarcasm, internet-native phrasing, stronger contrast.\n"
        "- CHAOTIC: absurd escalation, fast-turn twist, maximal meme energy.\n"
        "Infer mode from user tone. If unclear, default to SPICY.\n\n"
        "CORE COMEDY STANDARD:\n"
        "A meme is valid only if it has: setup, misdirection, and payoff.\n"
        "If the text reads like plain explanation, rewrite it before calling the tool.\n\n"
        "COMEDY PLAYBOOK (pick one per meme):\n"
        "- Expectation vs Reality\n"
        "- POV self-own\n"
        "- Historical event -> modern relatable pain\n"
        "- Overconfident plan -> immediate chaos\n"
        "- Glow-up / downfall timeline\n"
        "- Corporate tone for absurd situation\n\n"
        "EXTRA TEMPLATE BANK (rotate; do not reuse same template back-to-back):\n"
        "- Bro visited his own build\n"
        "- Task failed successfully\n"
        "- Me explaining vs me doing\n"
        "- Group project carried by one person\n"
        "- Patch notes nobody asked for\n"
        "- POV: universe sends a personalized L\n\n"
        "TEXT OVERLAY RULES:\n"
        "1) Keep overlay short and punchy: max 24 words per panel.\n"
        "2) Prefer 1-2 panels. Use 3 only if transformation is essential.\n"
        "3) Use specific nouns and culturally sharp details; avoid generic wording.\n"
        "4) No emoji in image text.\n"
        "5) Avoid safe statements like 'X happened' or 'I am sad'. Convert to a joke frame.\n"
        "6) Add tension words: 'me after', 'POV', 'meanwhile', 'bro thought', 'when you'.\n"
        "7) The last line must land as the punchline, not another setup.\n\n"
        "ANTI-BORING REWRITE FILTER (mandatory):\n"
        "- If caption contains bland verbs like 'realize', 'is', 'happened' without a twist, rewrite.\n"
        "- If joke can be predicted from first line, add misdirection in second line.\n"
        "- If both lines are same emotional tone, force contrast (confident -> cooked, hype -> despair).\n"
        "- Replace abstract wording with concrete detail (specific event, number, ritual, object).\n\n"
        "VISUAL DIRECTION RULES FOR THE IMAGE PROMPT:\n"
        "- Specify character emotion intensely (smug, cooked, panic, dead-inside, feral joy).\n"
        "- Specify shot type (close-up reaction, split screen, before/after, comic panel).\n"
        "- Specify meme typography: bold white uppercase with black stroke, high legibility.\n"
        "- Keep text in clean blocks with safe margins; do not cover faces.\n"
        "- Preserve factual integrity for historical content; do not invent fake numbers.\n\n"
        "SELF-CHECK BEFORE TOOL CALL:\n"
        "- Is there a clear comedic twist?\n"
        "- Would a person instantly get the joke in 2 seconds?\n"
        "- Is this quotable and repostable as a caption?\n"
        "- Is this different from the last generated meme template?\n"
        "If any answer is no, rewrite once and improve the hook.\n\n"
        "TOOL PROMPT CONSTRUCTION ORDER (always follow):\n"
        "1) Meme format + panel structure\n"
        "2) Character and facial expression\n"
        "3) Scene details and symbolism\n"
        "4) Exact overlay text lines in quotes\n"
        "5) Typography and safe-area placement\n"
        "6) Style keywords and quality constraints\n\n"
        "FINAL RESPONSE FORMAT AFTER GENERATION (for each image):\n"
        "- Meme Title + Markdown image link\n"
        "\n\n"
        "- A high-engagement caption in Gen Z tone\n"
        "- Relevant hashtags"
    )
    visualizer_agent = create_agent(text_llm, visualizer_tools, visualizer_prompt, "Visualizer")

    layout_prompt = (
        "You are a Technical Layout Polisher for meme performance. "
        "Use python_repl to compute aspect ratios, font scale ranges, text box bounds, panel splits, "
        "and safe-area placements so overlays stay readable on mobile feeds. "
        "Return concrete numbers the Visualizer can apply directly."
    )
    layout_agent = create_agent(text_llm, layout_tools, layout_prompt, "Layout_Calculator")
    
    research_tools_node = create_tool_node(research_tools, "Researcher_Tools")
    visualizer_tools_node = create_tool_node(visualizer_tools, "Visualizer_Tools")
    layout_tools_node = create_tool_node(layout_tools, "Layout_Calculator_Tools")

    workflow = StateGraph(AgentState)
    
    workflow.add_node("Supervisor", supervisor_node)
    
    workflow.add_node("Researcher", research_agent)
    workflow.add_node("Researcher_Tools", research_tools_node)
    
    workflow.add_node("Visualizer", visualizer_agent)
    workflow.add_node("Visualizer_Tools", visualizer_tools_node)
    
    workflow.add_node("Layout_Calculator", layout_agent)
    workflow.add_node("Layout_Calculator_Tools", layout_tools_node)

    workflow.add_edge(START, "Supervisor")
    
    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "Researcher": "Researcher",
            "Visualizer": "Visualizer",
            "Layout_Calculator": "Layout_Calculator",
            "FINISH": END
        }
    )
    
    def router_factory(agent_name):
        def route(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return f"{agent_name}_Tools"
            return "Supervisor"
        return route

    workflow.add_conditional_edges("Researcher", router_factory("Researcher"), {"Researcher_Tools": "Researcher_Tools", "Supervisor": "Supervisor"})
    workflow.add_conditional_edges("Visualizer", router_factory("Visualizer"), {"Visualizer_Tools": "Visualizer_Tools", "Supervisor": "Supervisor"})
    workflow.add_conditional_edges("Layout_Calculator", router_factory("Layout_Calculator"), {"Layout_Calculator_Tools": "Layout_Calculator_Tools", "Supervisor": "Supervisor"})
    
    workflow.add_edge("Researcher_Tools", "Researcher")
    workflow.add_edge("Visualizer_Tools", "Visualizer")
    workflow.add_edge("Layout_Calculator_Tools", "Layout_Calculator")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
