# Deployment & Hosting Plan: GeoShortMaker

> **Who this is for**: This document explains *why* moving to a hosted, containerized environment is valuable, and what concepts are involved. Specific technology choices (which cloud provider, which queue library, etc.) are intentionally left open — those decisions will be made when work actually begins.

---

## Why "Just Running It Locally" Has Limits

Right now, to generate a video, you run the script on your own machine. This works great for personal use, but it has hard limits:

- Only one person can use it at a time.
- Your computer has to stay on and busy for 5–15 minutes while a video renders.
- Sharing it with collaborators or users requires handing them the entire codebase and asking them to set up Python, Node.js, FFmpeg, and Playwright themselves.
- No record of past runs beyond the local `runs/` folder.

The goal of this plan is to move toward a hosted setup where a video can be requested via a simple interface, generated in the background on a server, and delivered when ready.

---

## 1. Containerization (Docker)

### The Concept

This project is **polyglot** — it uses multiple programming languages and runtimes:
- **Python** for the orchestration pipeline
- **Node.js** for the Cesium 3D renderer
- **FFmpeg** (a system-level binary) for video encoding
- **Playwright** (a headless Chrome browser) for capturing 3D renders

Getting all of these installed correctly on *your* machine took time. Doing it again on a server, and again on someone else's machine, is painful. Every server is slightly different — different OS, different versions, different paths.

**Docker** solves this by creating a "container": a self-contained package that includes the application *and everything it needs to run*, bundled together. You build the container once. It then runs identically on your laptop, a colleague's machine, or any cloud server in the world.

Think of it like the difference between giving someone a recipe (which they have to follow with their own ingredients and equipment) vs. giving them a sealed meal kit (which contains exactly what's needed and follows the same instructions every time).

### Why This Matters for This Project Specifically

The Playwright/Cesium renderer is the hardest dependency to set up. It launches a headless Chrome browser and drives it programmatically. This setup is notoriously environment-sensitive — it works on your machine but frequently breaks on others. Docker eliminates this problem by baking the browser and all its dependencies into the container image itself.

---

## 2. The Problem with Serverless (And Why We Can't Use It Here)

### The Concept

**Serverless** platforms (like AWS Lambda, Google Cloud Run, or Vercel) are popular for web applications. They work by running your code only when a request comes in, and shutting down immediately after. You pay only for what you use, and you never manage a server.

However, serverless has a critical limitation: **timeout**. Most serverless platforms limit a single function call to 15–60 seconds. Generating a geo video takes 5–15 minutes. Serverless is simply not an option here.

Additionally, serverless containers are typically limited in size (no room for a full Chromium browser installation) and don't support persistent file storage between requests.

---

## 3. The Right Pattern: Worker + Queue

### The Concept

The industry-standard solution for **long-running background jobs** is a **Worker + Queue** pattern. Here's how it works:

1. A lightweight **API** (a simple web server) receives the request: *"Make a video about Bangladesh."*
2. Instead of doing the work immediately, the API puts a **message** into a **queue**: *"Someone wants a Bangladesh video, here are the details."*
3. The API immediately responds to the user: *"Got it! Your video is queued."*
4. A separate **worker** process picks the message off the queue and does the actual work — running the full pipeline.
5. When the worker finishes, it saves the result somewhere accessible (cloud storage) and updates a status record (database) so the user can fetch their video.

This pattern is incredibly common. Almost every platform that does background processing uses it: YouTube video uploads, image resizing services, email senders, data export tools.

### Why Is This Pattern Used?

- **Non-blocking**: The user isn't waiting for 10 minutes. They get an immediate response.
- **Scalable**: If many videos are requested at once, you can run multiple workers in parallel.
- **Resilient**: If the worker crashes mid-render, the message can be "re-queued" and retried automatically.
- **Decoupled**: The API and the worker are completely independent. You can update the worker without touching the API, and vice versa.

---

## 4. Storage: The Disk Problem

### The Concept

Currently, the pipeline writes everything to a local `runs/` folder — raw frames, audio chunks, intermediate clips, and the final video. On a personal machine this is fine. On a cloud server, it creates two problems:

1. **Disk fills up**: A 60-second video can produce gigabytes of intermediate files. A server's disk is finite.
2. **Ephemeral servers**: Cloud servers (especially containerized ones) are often replaced or restarted. When that happens, the disk is wiped. Your runs folder disappears.

The solution is **object storage** — a cloud service specifically designed for storing files at arbitrary scale. AWS S3 and Google Cloud Storage are the industry standard. Instead of writing to `runs/<slug>/`, the pipeline writes to `s3://your-bucket/runs/<slug>/`. Files persist indefinitely, cost almost nothing to store, and can be shared via links.

---

## 5. Tracking Jobs: The Database

### The Concept

Once videos are generated in the background by a worker, you need a way to answer the question: *"Is my video ready yet?"*

The answer is a **database** that tracks job status. Every time a video is requested, a row is inserted:

| id | prompt | status | created_at | output_url |
|---|---|---|---|---|
| 1 | "Bangladesh geography" | `rendering` | 2025-04-16 | `null` |
| 2 | "Why is this river weird" | `done` | 2025-04-15 | `s3://...` |

The API can then serve a simple status endpoint: *"Check job 1 — still rendering. Check job 2 — here's your video link."*

This pattern — **async request → poll for status → receive result** — is how almost all professional video generation, export, and processing services work.

---

## Summary: The Full Hosted Architecture

```
User Request
     │
     ▼
  API Server (lightweight, fast)
     │  puts message in queue
     ▼
  Message Queue (job waiting line)
     │  worker picks up job
     ▼
  Worker Server (heavy, runs the full pipeline)
     │  writes output to
     ▼
  Cloud Storage (S3 / GCS)
     │  updates status in
     ▼
  Database (SQLite / PostgreSQL)
     │  user polls status, gets link
     ▼
  User receives video link
```

Every component here is a well-understood, battle-tested piece of infrastructure. The goal is not to invent anything new — it's to wire together proven tools in a proven pattern.

---

## Key Decisions Still to Make

These are the questions that will determine the specific tools chosen when implementation begins:

- **Cloud Provider** — AWS, GCP, DigitalOcean, Hetzner? (Each has tradeoffs in cost, simplicity, and regional availability.)
- **GPU availability** — Does the hosting environment have a GPU? The Cesium 3D renderer works much faster with GPU hardware acceleration. Without it, software rendering (SwiftShader) is used — slower but cheaper.
- **Queue technology** — Redis Queue (RQ) is simpler. Celery is more powerful. The right choice depends on scale.
- **Database** — SQLite is zero-maintenance for small scale. PostgreSQL handles concurrent writes better at larger scale.
