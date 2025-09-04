# Lexplain Backend Deployment Notes

## Cloud Run (Recommended)
- Build Docker image:
  ```sh
  docker build -t gcr.io/<your-project-id>/lexplain-backend .
  docker push gcr.io/<your-project-id>/lexplain-backend
  ```
- Deploy to Cloud Run:
  ```sh
  gcloud run deploy lexplain-backend \
    --image gcr.io/<your-project-id>/lexplain-backend \
    --platform managed \
    --region <region> \
    --allow-unauthenticated=false \
    --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/secrets/service-account.json
  ```
- Grant service account access to Firestore and GCS bucket.

## GCS Bucket Creation
```sh
gsutil mb -l <region> gs://lexplain-docs-<project-id>
gsutil iam ch serviceAccount:<service-account>@<project-id>.iam.gserviceaccount.com:objectAdmin gs://lexplain-docs-<project-id>
```

## Firestore Indexes
```sh
gcloud firestore indexes composite create --collection-group=chunks --field-config field-path=documentId,order=asc field-path=embedding,order=asc
```

## Firebase Hosting (Frontend)
- See frontend/README.md for setup.

## .env Setup
- Copy `.env.example` to `.env` and fill in credentials for local/dev/prod.

---
**Disclaimer:** This tool provides informational summaries only, not legal advice.
