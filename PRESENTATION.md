# FinPilot — Verifiable Planning Multi-Agent System for Adaptive Financial Planning

---

## 1. The Idea

**Domain:** FinTech — AI-Powered Personal Financial Planning & Advisory

**Problem Statement:**
- Traditional financial planning tools are **static** — they generate a plan once and leave users on their own
- Life is unpredictable: job loss, medical emergencies, market crashes, and family changes **invalidate** existing plans
- Existing solutions **lack transparency** — users can't see *why* a recommendation was made
- No tool continuously **monitors, verifies, and adapts** financial plans in real-time
- Users lose trust because AI-generated advice feels like a **black box**

---

## 2. The Concept

### Brief about the Idea
FinPilot is an **AI-powered multi-agent system** that doesn't just create financial plans — it **continuously monitors, verifies, and adapts** them in real-time. Five specialized AI agents collaborate transparently, showing users the complete reasoning behind every decision through an interactive visualization called **ReasonGraph**.

### Proposed Solution

**How it solves the problem:**
- A **Continuous Monitoring & Verification Loop (CMVL)** watches for market changes and life events, automatically triggering plan re-evaluation and adjustment
- Five specialized agents handle different aspects: orchestration, data retrieval, planning, verification, and execution — ensuring **no single point of failure** in decision-making
- **ReasonGraph** visualization makes every AI decision transparent and auditable

**How it differs from existing solutions:**

| Feature | Traditional Tools | Robo-Advisors | **FinPilot** |
|---------|:---:|:---:|:---:|
| Continuous monitoring | ❌ | ⚠️ Basic | ✅ CMVL |
| Multi-agent reasoning | ❌ | ❌ | ✅ 5 agents |
| Decision transparency | ❌ | ❌ | ✅ ReasonGraph |
| Life event adaptation | ❌ | ❌ | ✅ Real-time |
| Concurrent crisis handling | ❌ | ❌ | ✅ Multi-trigger |
| Verification loop | ❌ | ❌ | ✅ Built-in |

**Mechanism:**
1. User defines financial goals → **Orchestration Agent** parses and delegates
2. **Information Retrieval Agent** fetches real-time market data with multi-source validation
3. **Planning Agent** uses Thought-of-Search (ToS) algorithm to explore 5+ strategic paths
4. **Verification Agent** validates constraints, safety margins, and regulatory compliance
5. **Execution Agent** implements the plan with tax optimization
6. **CMVL** continuously monitors for triggers → loops back to step 2 when needed

### USP (Unique Selling Proposition)
> **"The only financial planning system where you can see, verify, and trust every decision the AI makes — and that adapts to your life in real-time."**

- 🔍 **Verifiable** — Every decision is traceable through ReasonGraph
- 🔄 **Adaptive** — CMVL responds to life events + market changes simultaneously
- 🤖 **Multi-Agent** — 5 specialized agents, not a single monolithic AI
- 🛡️ **Safe** — Circuit breakers, verification loops, and rollback capabilities

### List of Features

**Core:**
- ✅ Multi-agent financial plan generation with 5+ strategy paths
- ✅ Interactive ReasonGraph visualization (React + D3.js)
- ✅ Real-time market data integration (Alpha Vantage, Yahoo Finance)
- ✅ Continuous Monitoring & Verification Loop (CMVL)
- ✅ Thought-of-Search algorithm for guided plan exploration
- ✅ Circuit breaker patterns for fault tolerance

**Intelligence:**
- ✅ Market volatility detection and trigger classification
- ✅ Life event handling (job loss, medical emergency, family changes)
- ✅ Concurrent multi-trigger crisis management
- ✅ Tax optimization and regulatory compliance checking
- ✅ Risk-adjusted return optimization

**User Experience:**
- ✅ Conversational AI chatbot for natural language goal input
- ✅ Before/after plan comparison with impact analysis
- ✅ Interactive decision tree exploration with filtering
- ✅ Real-time agent status monitoring dashboard

---

## 3. Technical Visualization

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)               │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Dashboard │  │ AI Chatbot   │  │ ReasonGraph (D3)  │  │
│  └─────┬─────┘  └──────┬───────┘  └────────┬──────────┘  │
│        └───────────────┼────────────────────┘             │
│                        │ REST API + WebSocket             │
├────────────────────────┼────────────────────────────────-─┤
│                   BACKEND (FastAPI)                       │
│                        │                                  │
│  ┌─────────────────────▼──────────────────────────────┐  │
│  │           ORCHESTRATION AGENT (OA)                  │  │
│  │     Session Mgmt · Routing · Circuit Breakers       │  │
│  └──┬──────────┬──────────────┬───────────────┬───────┘  │
│     │          │              │               │          │
│     ▼          ▼              ▼               ▼          │
│  ┌──────┐  ┌──────┐     ┌──────┐        ┌──────┐        │
│  │ IRA  │  │  PA  │     │  VA  │        │  EA  │        │
│  │Market│  │Plan +│     │Verify│        │ Exec │        │
│  │ Data │  │ ToS  │     │+CMVL │        │+ Tax │        │
│  └──┬───┘  └──┬───┘     └──┬───┘        └──┬───┘        │
│     │         │            │               │             │
│     └─────────┴──────┬─────┴───────────────┘             │
│                      │                                    │
│              ┌───────▼────────┐                           │
│              │  Agent Comms   │                           │
│              │  (Correlation  │                           │
│              │   ID + Redis)  │                           │
│              └────────────────┘                           │
├──────────────────────────────────────────────────────────┤
│  EXTERNAL: Market APIs · Redis Cache · PostgreSQL        │
└──────────────────────────────────────────────────────────┘
```

### Process Flow / Use-case Diagram — CMVL Workflow

```
User Goal Input
      │
      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│    OA    │───▶│   IRA    │───▶│    PA    │───▶│    VA    │
│ Parse &  │    │ Fetch    │    │ Generate │    │ Verify & │
│ Delegate │    │ Market   │    │ 5+ Plans │    │ Validate │
└──────────┘    │ Context  │    │ via ToS  │    └────┬─────┘
                └──────────┘    └──────────┘         │
                                                     │
                     ┌──────────────────┐      Pass? │
                     │                  │◀─────Yes───┘
                     │    EA: Execute   │       │
                     │    Plan + Tax    │    No │
                     │    Optimization  │       │
                     └───────┬──────────┘       ▼
                             │           ┌──────────┐
                             │           │ Re-Plan  │
                     ┌───────▼──────┐    │ with new │
                     │    CMVL      │    │ constr.  │
                     │  Continuous  │    └──────────┘
                     │  Monitoring  │
                     └───────┬──────┘
                             │
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
           Market Event  Life Event  Schedule
           (crash,       (job loss,   (quarterly
            spike)       medical)     review)
                 │           │           │
                 └───────────┼───────────┘
                             ▼
                     Re-trigger OA
                     (Loop back ↑)
```

---

## 4. Execution & Tools

### Implementation & Outcome Analysis

**Execution Summary:**
- Multi-agent architecture with 5 specialized Python agents communicating via structured messages with correlation IDs
- React frontend with D3.js-powered ReasonGraph for decision transparency
- CMVL system that monitors market volatility, life events, and scheduled reviews
- Thought-of-Search algorithm explores 5+ financial strategies using hybrid BFS/DFS

**Feasibility:**
- ✅ **Core system built and functional** — all 5 agents operational
- ✅ **Frontend live and deployed** via Vercel
- ✅ **CI/CD pipeline** active with GitHub Actions
- ✅ **Pydantic data contracts** ensure inter-agent type safety
- ✅ **Market APIs** integrated (Alpha Vantage, Yahoo Finance)
- ⚠️ Database persistence and auth are in-progress (currently in-memory)

**Intended Impact:**
- Empower individuals to make **informed, transparent** financial decisions
- Reduce financial planning anxiety with **continuous adaptation**
- Democratize access to **advisor-level** financial intelligence
- Increase user trust through **verifiable AI reasoning**

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, TypeScript, Vite, D3.js, Radix UI, Recharts, Tailwind CSS |
| **Backend** | Python 3.11, FastAPI, Pydantic v2, LangChain |
| **AI/ML** | Ollama (Local LLM), scikit-learn, NetworkX, NumPy, pandas |
| **Data** | Alpha Vantage API, Yahoo Finance, Redis (cache), PostgreSQL |
| **DevOps** | GitHub Actions CI/CD, Vercel (frontend), Docker |
| **Testing** | pytest, Playwright (E2E), bandit (security) |
| **Visualization** | D3.js (ReasonGraph), Recharts (charts), Mermaid (docs) |

### Research Sources

1. **Tree of Thoughts (ToT)** — Yao et al., 2023 — *"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"* — Basis for our Thought-of-Search algorithm
2. **Multi-Agent Systems in Finance** — IEEE, 2022 — *"A Survey of Multi-Agent Systems for Financial Applications"*
3. **Retrieval-Augmented Generation (RAG)** — Lewis et al., 2020 — *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*
4. **Circuit Breaker Pattern** — Microsoft Azure Architecture Center — Resilience patterns for distributed systems
5. **Continuous Verification** — Google SRE Book — Monitoring and verification best practices adapted for financial AI
6. **Constraint Satisfaction in Planning** — Russell & Norvig, *AI: A Modern Approach* — Foundation for verification agent logic
7. **Alpha Vantage API Documentation** — Real-time and historical market data integration
8. **Yahoo Finance API** — Market data retrieval and validation

---

> **FinPilot** — *Because your financial plan should be as dynamic as your life.*
