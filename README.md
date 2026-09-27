# AaaS_iThome

[Data](https://www.kaggle.com/datasets/beamhonor0911/youtube-trending-video-statistics-analytics?resource=download)

## Layout

```
backend/     Python: the agent, sandboxes, datasources, config, skills, tests
frontend/    (later)
data/        the CSVs (git-ignored)
articles/    the article drafts
docker-compose.yml, .env   Postgres, for both sides
```

## Backend

Run from `backend/`; secrets are in the repo-root `.env`.

```bash
cd backend
uv sync
uv run pytest
uv run --env-file ../.env python scripts/ask.py "Which category gets the most views?"
```
