# Pokédex Z – Profesor Z RAG

Agente RAG sobre la guía oficial de Pokémon Z, desplegado en **Google Cloud Run** con **Terraform** y **GitHub Actions**. Backend FastAPI + frontend React (Vite). Embeddings con **Gemini Embedding 001**, índice **FAISS** persistido en **Cloud Storage**, LLM **Gemini 2.5 Flash Lite**.

## Arquitectura

```
                ┌──────────────────────┐
                │ GitHub Actions (CI)  │
                │ build + tf apply     │
                └──────────┬───────────┘
                           │ WIF (sin keys)
                           ▼
 ┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐
 │ Artifact Reg │──▶│ Cloud Run (api)  │──▶│ Secret Manager     │
 │ (imagen)     │   │ FastAPI + React  │   │ GOOGLE_API_KEY     │
 └──────────────┘   └────────┬─────────┘   └────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Cloud Storage    │
                    │  - GuiaPokemonZ  │
                    │  - index.faiss   │
                    │  - chunks.pkl    │
                    └──────────────────┘
```

El flujo de RAG:

1. Un job offline (`indexer/build_index.py`) descarga el PDF del bucket, lo trocea, llama a Gemini Embedding y guarda `index.faiss` + `chunks.pkl` en GCS.
2. Cloud Run, al arrancar, descarga esos dos blobs y carga FAISS en memoria (no se recalculan embeddings en cada cold start).
3. En query: el backend embebe la pregunta con Gemini, hace búsqueda híbrida FAISS + BM25, y manda los top-K chunks a Gemini 2.5 Flash con el prompt del Profesor Z.

> Gemini Embedding **no lee del bucket directamente** — es la API la que recibe texto y devuelve vectores; tu código orquesta descarga / chunking / embed / persistencia.

## Estructura

```
backend/   FastAPI + indexer + Dockerfile
frontend/  React + Vite + TS
terraform/ módulos (gcs, artifact_registry, cloud_run, iam, secret_manager, workload_identity) + env dev
.github/workflows/  ci.yml, cd.yml, reindex.yml
```

## Bootstrap (una sola vez)

Requisitos: `gcloud`, `terraform >= 1.9`, una cuenta de Google AI Studio para la API key.

```bash
# 1. Autenticación local
gcloud auth login
gcloud auth application-default login

# 2. Crear proyecto (o reusar uno existente)
PROJECT_ID="mi-pokez-dev"
gcloud projects create $PROJECT_ID
gcloud beta billing projects link $PROJECT_ID --billing-account=XXXXXX-XXXXXX-XXXXXX

# 3. Habilitar APIs y crear bucket de tfstate
cd terraform
PROJECT_ID=$PROJECT_ID REGION=us-central1 ./bootstrap.sh

# 4. Editar terraform/envs/dev/backend.tf y poner el bucket que imprime el script

# 5. Subir el PDF al bucket
BUCKET="${PROJECT_ID}-pokez-data"
gcloud storage buckets create gs://$BUCKET --location=us-central1 --uniform-bucket-level-access
gcloud storage cp ../GuiaPokemonZ.pdf gs://$BUCKET/GuiaPokemonZ.pdf

# 6. Primer terraform apply (crea Artifact Registry, IAM, WIF, Cloud Run con imagen placeholder)
cp envs/dev/terraform.tfvars.example envs/dev/terraform.tfvars
# editar valores
cd envs/dev
terraform init
terraform apply
```

> El primer `apply` puede fallar al desplegar Cloud Run si la imagen aún no existe. Una vez el repo esté creado, deja que GitHub Actions construya y empuje la imagen y vuelve a aplicar (o usa una imagen pública placeholder en `container_image` para el primer apply).

### Construir el índice por primera vez

Tras subir el PDF al bucket, ejecuta el indexer en local o lanza el workflow `Reindex`:

```bash
cd backend
export GCP_PROJECT=$PROJECT_ID
export GCS_BUCKET=$BUCKET
export GOOGLE_API_KEY="AIza..."
python -m indexer.build_index
```

## Configurar GitHub Actions

En `Settings → Secrets and variables → Actions`, crea:

| Secret | Valor |
|---|---|
| `GCP_PROJECT_ID` | tu project id |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | output `workload_identity_provider` de Terraform |
| `GCP_CI_SERVICE_ACCOUNT` | output `ci_service_account` de Terraform |
| `GOOGLE_API_KEY` | clave de Gemini (también va a Secret Manager por TF) |
| `GCS_BUCKET` | output `bucket_name` |

En cada push a `main` con cambios en `backend/`, `frontend/` o `terraform/`, el workflow `cd.yml` construye la imagen, la sube a Artifact Registry y aplica Terraform actualizando Cloud Run.

## Desarrollo local

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # (PowerShell: .venv\Scripts\Activate)
pip install -r requirements.txt

export GCP_PROJECT=mi-pokez-dev
export GCS_BUCKET=mi-pokez-dev-pokez-data
export GOOGLE_API_KEY="AIza..."

uvicorn app.main:app --reload --port 8080
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173 con proxy /api → :8080
```

### Reindexar

```bash
cd backend
python -m indexer.build_index
```

## Endpoints

- `GET /api/health` → `{ status, index_loaded, chunks }`
- `POST /api/chat` → `{ message, history: [{role, content}] }` ⇒ `{ answer, sources, elapsed_ms }`
- `GET /` → frontend React (build estático servido por FastAPI)

## Tradeoffs y notas

- **FAISS en memoria** = simple y barato pero el índice tiene que caber en RAM. Para esta guía (~50MB PDF, ~5k chunks) sobra con `2Gi` en Cloud Run.
- **Cold start ~10s** porque hay que descargar el índice de GCS. Se mitiga con `min_instances=1` (cuesta dinero) o aceptando el delay del primer request.
- La API key de Gemini está en **Secret Manager** y se monta en Cloud Run como variable de entorno por `value_source.secret_key_ref`. Para producción seria, usar SA ADC contra Vertex AI en lugar de API key.
- **WIF** evita JSON keys: GitHub OIDC ⇄ pool restringido por `assertion.repository`.
- Estado de Terraform en GCS con versionado activo. El `bootstrap.sh` lo crea idempotente.
