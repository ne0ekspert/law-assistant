# law-assistant (한국어)

한국 법령 코퍼스(`law/`)를 대상으로 하는 FAISS 기반 RAG(검색 증강 생성) 에이전트입니다.
임베딩과 대화 모델로 Ollama, OpenAI, Anthropic를 지원합니다.

## 빠른 시작

- 환경 프리셋 중 하나를 `.env`로 복사:
  - `cp .env.ollama .env` (로컬 모델)
  - `cp .env.openai .env` (OpenAI 전용)
  - `cp .env.anthropic .env` (Anthropic 채팅 + 로컬 임베딩)
- Python 3.10+ 설치 확인
- 의존성 설치: `pip install -e .` (또는 `pip install -r requirements.txt`)

## 인덱스 빌드

- 기본값: Ollama 임베딩 `nomic-embed-text` 사용
  - `law-assistant index --chunk-size 800 --chunk-overlap 100`
- OpenAI 임베딩 예시:
  - `EMBEDDINGS_PROVIDER=openai EMBEDDINGS_MODEL=text-embedding-3-small law-assistant index`

## RAG 질의

- 기본(환경의 `CHAT_MODEL`)으로 Ollama 채팅:
  - `law-assistant query "행정절차법의 처분 기준은?"`
- OpenAI 채팅:
  - `CHAT_PROVIDER=openai CHAT_MODEL=gpt-5-mini law-assistant query "부칙 적용 기준은?"`
- Anthropic 채팅:
  - `CHAT_PROVIDER=anthropic CHAT_MODEL=claude-3-5-sonnet-20240620 law-assistant query "형법 총칙 요건?"`

## 참고 사항

- 임베딩 제공자: `ollama`, `openai`, `anthropic` (Anthropic 임베딩은 계정/플랜에 따라 `/v1/embeddings` 제공이 없을 수 있음)
- Ollama는 임베딩 가능 모델이 필요합니다. 예: `ollama pull nomic-embed-text`
- 인덱스 파일은 `data/`에 저장됩니다 (`index.faiss`, `meta.json`).

## 에이전트 채팅 루프

- 먼저 인덱스를 빌드하세요(위 참조).
- 실행: `python agent.py` (인덱스가 없으면 자동 빌드)
- 자세한 로그: `python agent.py --verbose` (검색/LLM 타이밍 및 툴 호출 표시)
- 명령어: `:exit`, `:reset`, `:topk <int>`

## 툴 호출(툴 사용)

- `CHAT_PROVIDER=openai` 또는 `anthropic`일 때, 모델은 `retrieve(query, top_k)` 툴을 호출해 법령 구절을 가져올 수 있습니다. 이 방식은 프롬프트에 긴 컨텍스트를 직접 삽입하는 대신 필요한 시점에 조회합니다.
- 기타 제공자(예: `ollama`)는 툴 호출을 지원하지 않으므로 조회된 컨텍스트를 사용자 메시지에 삽입하는 방식으로 동작합니다.
- 자세한 단계별 로그는 `--verbose` 플래그로 확인하세요.

## 면책 조항

- 본 프로젝트 및 제공되는 에이전트/스크립트/모델 응답은 법률 자문을 제공하기 위한 것이 아니며, 어떠한 법적 판단이나 결정의 근거로 사용되어서는 안 됩니다.
- 법령·판례·행정규칙 등은 수시로 개정·변경될 수 있으며, 본 저장소의 데이터(`law/` 서브모듈 포함)와 생성된 색인 정보의 정확성·완전성·최신성을 보장하지 않습니다.
- LLM의 특성상 환각(hallucination)이나 오해의 소지가 있는 답변이 포함될 수 있습니다. 반드시 원문과 공식 출처를 직접 확인하시기 바랍니다.
- 본 프로젝트의 사용으로 인한 어떠한 손해에 대해서도 책임을 지지 않습니다. 모든 책임은 사용자에게 있습니다.
- 실제 법률 문제는 변호사 등 자격 있는 전문가와 상담하시기 바랍니다.
