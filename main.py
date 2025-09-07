import os, asyncio, datetime as dt, json, re, html, logging
from typing import List, Literal, Optional, TypedDict

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("risk-agent")

MAX_GUESS_TRIES = int(os.getenv("MAX_GUESS_TRIES", "3"))

# ---------------- LLM (LangChain + OpenAI) ----------------
from langchain_openai import ChatOpenAI

llm_temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not os.getenv("OPENAI_API_KEY"):
    log.warning("OPENAI_API_KEY is not set. Model calls will fail.")

LLM = ChatOpenAI(model=MODEL_NAME, temperature=llm_temp)

# ---------------- Models (STRICT 4-key output) ----------------
class IntentModel(BaseModel):
    model_config = ConfigDict(title="IntentModelV5", extra="forbid")
    entity: str
    dimension: Literal["governance", "environment", "social"]

class EvidenceChunkModel(BaseModel):
    model_config = ConfigDict(title="EvidenceChunkModelV5", extra="forbid")
    source: Literal["sec", "news", "wikipedia", "company", "web"]
    url: str
    published_at: Optional[str] = None
    snippet: str

class AnalysisOutputStrict(BaseModel):
    model_config = ConfigDict(title="AnalysisOutputStrictV5", extra="forbid")
    entity: str
    dimension: Literal["governance", "environment", "social"]
    risks: List[str] = Field(default_factory=list)
    status: Literal["analysis_complete", "no_data_found", "fallback_used"] = "analysis_complete"

class PipelineState(TypedDict):
    query: str
    intent: Optional[IntentModel]
    raw_evidence: List[EvidenceChunkModel]
    consolidated: List[EvidenceChunkModel]
    output: Optional[AnalysisOutputStrict]
    trace: List[str]

# ---------------- Prompts + structured output ----------------
from langchain.prompts import PromptTemplate

# Use OpenAI tool-calling via structured outputs to avoid JSON parse errors
STRUCT_INTENT = LLM.with_structured_output(IntentModel)
STRUCT_FINAL  = LLM.with_structured_output(AnalysisOutputStrict)

INTENT_PROMPT = PromptTemplate(
    template=(
        "You are an intake triage assistant.\n"
        "- Extract COMPANY/ENTITY (not an industry bucket) and ESG DIMENSION.\n"
        "- If the query contains 'privacy', map dimension to 'social'.\n\n"
        "USER QUERY:\n{user_query}\n"
    ),
    input_variables=["user_query"],
)

EVIDENCE_CONSOLIDATION_PROMPT = PromptTemplate.from_template(
    "You are a research synthesizer for {entity} ({dimension}).\n"
    "Keep only relevant, recent items and compress each into 1–2 sentence snippets (no HTML).\n\n"
    "INPUT EVIDENCE (JSON):\n{evidence_json}\n\n"
    "Return a JSON array with items of shape:\n"
    '[{\\"source\\":\\"sec|news|wikipedia|company|web\\",\\"url\\":\\"...\\",\\"published_at\\":\\"YYYY-MM-DD\\",\\"snippet\\":\\"...\\"}]'
)

FINAL_PROMPT = PromptTemplate(
    template=(
        "You are a senior {dimension} risk analyst.\n"
        "Use ONLY the evidence to extract short, concrete risks for {entity}.\n"
        "Each risk must be a plain sentence—no HTML/links/markup.\n\n"
        "EVIDENCE JSON:\n{evidence_json}\n"
    ),
    input_variables=["entity","dimension","evidence_json"],
)

GUESS_PROMPT = PromptTemplate.from_template(
    "You are a {dimension} risk analyst.\n"
    "Evidence was insufficient. Using your knowledge and the context below, provide 2–5 plausible {dimension} risks "
    "that {entity} may be facing now or in the future. NEVER fewer than 2.\n"
    "Each risk must be a plain sentence (no HTML/URLs/markup), <= 18 words.\n\n"
    "Original user query:\n{user_query}\n\n"
    "Context (snippets):\n{context_hint}"
)

# ---------------- External tools (HTTP) ----------------
import httpx

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

async def wikipedia_search(entity: str) -> List[EvidenceChunkModel]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action":"query","list":"search","srsearch":entity,"format":"json"}
    out=[]
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        for hit in data.get("query",{}).get("search",[])[:5]:
            title = (hit.get("title") or "").replace(" ","_")
            href = f"https://en.wikipedia.org/wiki/{title}"
            out.append(EvidenceChunkModel(source="wikipedia", url=href, snippet=hit.get("snippet") or ""))
    except Exception as e:
        log.warning(f"wikipedia_search error: {e!r}")
    return out

async def newsapi_search(entity: str, dimension: str) -> List[EvidenceChunkModel]:
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    q_dim = {
        "social": "privacy OR data protection OR content moderation OR labor OR safety",
        "governance": "governance OR board OR audit OR compliance",
        "environment": "emissions OR sustainability OR pollution OR climate"
    }[dimension]
    q = f'"{entity}" {q_dim}'
    params = {"q": q, "language":"en", "pageSize": 10, "sortBy":"publishedAt", "apiKey": NEWSAPI_KEY}
    out=[]
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        for a in (data.get("articles") or [])[:6]:
            out.append(EvidenceChunkModel(
                source="news",
                url=a.get("url") or "",
                published_at=(a.get("publishedAt") or "")[:10] or None,
                snippet=((a.get("title") or "") + ". " + (a.get("description") or ""))[:280]
            ))
    except Exception as e:
        log.warning(f"newsapi_search error: {e!r}")
    return out

async def company_site(entity: str) -> List[EvidenceChunkModel]:
    # lightweight heuristic (HTTP-only, no hard-coded brands)
    domain_guess = re.sub(r"[^a-zA-Z0-9]", "", entity).lower()
    urls = [
        f"https://{domain_guess}.com/investors/corporate-governance",
        f"https://{domain_guess}.com/corporate-governance",
        f"https://{domain_guess}.com/esg",
        f"https://{domain_guess}.com/sustainability"
    ]
    out=[]
    try:
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent":"DueDiligenceBot/1.0"}) as client:
            for u in urls:
                try:
                    r = await client.get(u)
                    if r.status_code==200 and len(r.text)>500:
                        out.append(EvidenceChunkModel(source="company", url=u, snippet="Company page discovered."))
                        break
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"company_site error: {e!r}")
    return out

async def sec_filings(entity: str) -> List[EvidenceChunkModel]:
    # Placeholder for EDGAR/CIK integration
    return []

# ---------------- Sanitizers ----------------
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
WS_RE  = re.compile(r"\s+")

def clean_sentence(s: str) -> str:
    s = (s or "")
    s = s.replace("\\n"," ").replace("\\t"," ").replace('\\"','"').replace("\\/","/")
    s = html.unescape(s)
    s = TAG_RE.sub("", s)
    s = URL_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()
    return s[:200]

def sanitize_output(o: AnalysisOutputStrict) -> AnalysisOutputStrict:
    o.risks = [clean_sentence(x) for x in (o.risks or []) if clean_sentence(x)]
    o.entity = clean_sentence(o.entity)
    return o

# ---------------- LangGraph nodes ----------------
from langgraph.graph import StateGraph, END
from pydantic import BaseModel as _PydBase

def _read_status(out) -> Optional[str]:
    if out is None: return None
    if isinstance(out, dict): return out.get("status")
    if isinstance(out, _PydBase):
        try: return out.status
        except Exception: return None
    return getattr(out, "status", None)

async def node_intent(state: PipelineState) -> PipelineState:
    try:
        res: IntentModel = await (INTENT_PROMPT | STRUCT_INTENT).ainvoke({"user_query": state["query"]})
    except Exception as e:
        # Minimal fallback so pipeline can proceed
        state.setdefault("trace", []).append(f"intent_error:{e!r}")
        res = IntentModel(entity=state["query"], dimension="social")
    state["intent"] = res
    state["trace"].append(f"intent:{res.model_dump_json()}")
    return state

async def node_retrieve(state: PipelineState) -> PipelineState:
    ent, dim = state["intent"].entity, state["intent"].dimension
    results = await asyncio.gather(
        wikipedia_search(ent),
        newsapi_search(ent, dim),
        sec_filings(ent),
        company_site(ent),
        return_exceptions=True
    )
    raw: List[EvidenceChunkModel] = []
    for r in results:
        if isinstance(r, list):
            raw.extend(r)
    state["raw_evidence"] = raw
    state["trace"].append(f"retrieved:{len(raw)}")
    return state

def is_weak(state: PipelineState) -> bool:
    if not state["raw_evidence"]: return True
    recent_cut = dt.date.today() - dt.timedelta(days=365*3)
    fresh = 0
    for e in state["raw_evidence"]:
        try:
            if e.published_at and dt.date.fromisoformat(e.published_at) >= recent_cut:
                fresh += 1
        except Exception:
            pass
    return len(state["raw_evidence"]) < 3 or fresh == 0

async def node_consolidate(state: PipelineState) -> PipelineState:
    intent = state["intent"]
    payload = {
        "entity": intent.entity,
        "dimension": intent.dimension,
        "evidence_json": [e.model_dump() for e in state["raw_evidence"]],
    }
    txt = await (EVIDENCE_CONSOLIDATION_PROMPT | LLM).ainvoke(payload)
    try:
        consolidated = [EvidenceChunkModel(**x) for x in json.loads(txt.content)]
    except Exception:
        consolidated = state["raw_evidence"][:8]
    state["consolidated"] = consolidated
    state["trace"].append(f"consolidated:{len(consolidated)}")
    return state

async def node_analyze(state: PipelineState) -> PipelineState:
    """Always run LLM to extract risks from tool evidence using structured output."""
    intent = state["intent"]
    evidence_json = json.dumps([e.model_dump() for e in state["consolidated"]], ensure_ascii=False)
    payload = {"entity": intent.entity, "dimension": intent.dimension, "evidence_json": evidence_json}
    try:
        result: AnalysisOutputStrict = await (FINAL_PROMPT | STRUCT_FINAL).ainvoke(payload)
    except Exception as e:
        state["trace"].append(f"analyze_error:{e!r}")
        result = AnalysisOutputStrict(entity=intent.entity, dimension=intent.dimension, risks=[], status="no_data_found")
    state["output"] = sanitize_output(result)
    state["trace"].append(f"analyzed:{state['output'].status}")
    return state

async def node_llm_guess(state: PipelineState) -> PipelineState:
    """If extraction fails or <2 risks, ask LLM again to write risks from collected context (no code-made generics)."""
    intent = state["intent"]
    tries, result = 0, None
    snippets = [e.snippet for e in state.get("consolidated", [])][:4]
    context_hint = " ".join(snippets) if snippets else ""

    while tries < MAX_GUESS_TRIES:
        tries += 1
        payload = {"entity": intent.entity, "dimension": intent.dimension, "user_query": state["query"], "context_hint": context_hint}
        try:
            candidate: AnalysisOutputStrict = await (GUESS_PROMPT | STRUCT_FINAL).ainvoke(payload)
            candidate = sanitize_output(candidate)
            if candidate.status == "analysis_complete" and len(candidate.risks) >= 2:
                result = candidate
                break
        except Exception as e:
            state["trace"].append(f"guess_error_try{tries}:{e!r}")
            continue

    if result is None:
        result = AnalysisOutputStrict(entity=intent.entity, dimension=intent.dimension, risks=[], status="no_data_found")

    state["output"] = result
    state["trace"].append(f"llm_guess:{result.status} tries={tries}")
    return state

async def node_fallback(state: PipelineState) -> PipelineState:
    # Optional enrichment: broaden news on first token
    ent = state["intent"].entity.split()[0]
    extra = await newsapi_search(ent, state["intent"].dimension)
    state["raw_evidence"].extend(extra)
    state["trace"].append(f"fallback_added:{len(extra)}")
    return state

# ---------------- Build Graph ----------------
from langgraph.graph import StateGraph, END

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("intent",      node_intent)
    g.add_node("retrieve",    node_retrieve)
    g.add_node("consolidate", node_consolidate)
    g.add_node("analyze",     node_analyze)
    g.add_node("fallback",    node_fallback)
    g.add_node("guess",       node_llm_guess)

    g.set_entry_point("intent")
    g.add_edge("intent","retrieve")
    g.add_edge("retrieve","consolidate")

    def cond_after_consolidate(state: PipelineState):
        return "fallback" if is_weak(state) else "analyze"
    g.add_conditional_edges("consolidate", cond_after_consolidate, {"fallback":"fallback","analyze":"analyze"})
    g.add_edge("fallback","analyze")

    def cond_after_analyze(state: PipelineState):
        out = state.get("output")
        status = _read_status(out)
        risks = out.risks if isinstance(out, AnalysisOutputStrict) else (out.get("risks") if isinstance(out, dict) else [])
        return "guess" if (status != "analysis_complete" or len(risks) < 2) else "done"
    g.add_conditional_edges("analyze", cond_after_analyze, {"guess":"guess","done":END})

    g.add_edge("guess", END)
    return g.compile()

GRAPH = build_graph()

# ---------------- FastAPI ----------------
app = FastAPI(title="Risk Analyzer (Tools + LLM, 4-key JSON)", version="5.0.0")

class QueryIn(BaseModel):
    model_config = ConfigDict(title="QueryInV5", extra="forbid")
    query: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/analyze", response_model=AnalysisOutputStrict, response_model_exclude_none=True)
async def analyze(q: QueryIn) -> AnalysisOutputStrict:
    state: PipelineState = {"query": q.query, "intent": None, "raw_evidence": [], "consolidated": [], "output": None, "trace": []}
    out_state = await GRAPH.ainvoke(state)

    out: AnalysisOutputStrict = out_state["output"]  # type: ignore
    if (out is None) or (out.status != "analysis_complete") or (len(out.risks) < 2):
        out_state = await node_llm_guess(out_state)
        out = out_state["output"]  # type: ignore

    out = sanitize_output(out)
    log.info(f"RESULT: {out.model_dump()}")
    return out
