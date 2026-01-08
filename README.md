# PRAGMA41


Backend code: main.py
Requirements: requirements.txt
Model + UI copied into place: model.onnx, index.html
If you want to run it:

1. python -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. uvicorn app.main:app --reload --port 8080

Database (Postgres + Alembic):

1. Set `DATABASE_URL` in `.env`
2. alembic upgrade head

Let me know if you want the endpoint changed, a different model, or NMS behavior tuned.
