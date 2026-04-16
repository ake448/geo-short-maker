# Refactor Plan: Geo-Short-Maker

> **Who this is for**: This document describes *why* certain changes are valuable, not just *what* to change. The goal is to refactor this codebase by following industry best practices — things the pros use every day in larger codebases.
>
> Implementation details here are intentionally high-level. The specifics will be decided when work begins.

---

## A Note on What "Refactoring" Means

Your pipeline works. That's a real achievement. Refactoring is not about fixing broken things — it's about making working code *easier to change*, *easier to understand*, and *harder to accidentally break* as it grows.

Think of it like this: a script written in a single notebook works fine for one person. But the moment you want to hand it to someone else, run it in the cloud, or add a new feature six months from now, the "just make it work" approach starts costing you time. Refactoring is your investment into the future.

---

## 1. Domain-Driven Design (DDD)

### The Concept

Right now, different concerns are mixed together throughout the codebase. For example, there is logic for *what the video is about* (the geography facts, the script) sitting right next to logic for *how to render it* (FFmpeg commands, tile downloads). 

**Domain-Driven Design (DDD)** is a way of thinking about code that says: *group code by the real-world concept it represents, not by the technical job it does.*

In this codebase there are a few clear "real world concepts" (called **domains** or **bounded contexts**):

- **The Story** — What is this video about? What are the beats? What does the narrator say? This is the creative domain. It doesn't know or care about FFmpeg.
- **The Geography** — Where on earth is this? What are the boundaries? What do the tiles look like? This is the GIS/mapping domain.
- **The Media** — How do we turn the story + geography into actual pixels and audio? This is the infrastructure domain. It's technical and messy.
- **The Coordinator** — Something needs to orchestrate all three of these. That's the application layer (currently `runner.py`).

### Why Does This Matter?

When these concerns are mixed, a small change in one place can break something unexpected in another. Separating them means:
- You can change how the video is rendered (switch from FFmpeg to another tool) without touching the script logic.
- You can add a new b-roll type without worrying about the audio code.
- A new developer (or you, six months later) can understand what each file is responsible for just by looking at its name.

### What This Looks Like in Practice

You create formal objects representing your business concepts. A `Beat` is not just a Python dictionary — it's a class with rules. A `Region` is not just a string — it has a lat/lon, a bounding box, and a name. When these objects are passed around, you *know* what data they carry and what they can do.

---

## 2. Configuration & Pydantic

### The Concept

This project has a lot of configuration: API keys, file paths, model names, timeouts. Right now, all of that lives in `config.py` as loose global variables that get loaded from a `.env` file using a custom hand-rolled parser.

The problem: **nothing validates this configuration at startup.** If you forget to set `GEMINI_API_KEY`, the pipeline runs fine until the very moment it makes its first API call — and then crashes deep inside the stack with a confusing error.

**Pydantic** is a Python library that makes this dramatically better. You define a `Settings` class that describes every configuration value, its type, whether it's required, and what the default is. Pydantic reads the `.env` file and validates all of it the moment your program starts. If something is missing or wrong, you get a clear, readable error *immediately*, not ten minutes into a render.

### Why Does This Matter?

- **Fail fast, fail clearly.** A missing API key error at startup is 10x easier to debug than a crash halfway through rendering beat 7.
- **Type safety.** `PORT` is an integer. `GEMINI_API_KEY` is a non-empty string. Pydantic enforces this automatically.
- **Documentation by existence.** Your `Settings` class becomes a living document of every config value the project supports.

### What This Looks Like in Practice

Instead of a bunch of `os.environ.get("THING", "")` calls scattered across multiple files, there is one single `Settings` object imported everywhere. You add a field, you get it everywhere auto-validated. Simple.

---

## 3. Concurrency & Performance

### The Concept

The current pipeline does almost everything *one step at a time* — it downloads one tile, then the next, then the next. It renders one clip, then waits for it to finish before starting the next. This is called **serial** or **synchronous** execution.

Modern computers have multiple CPU cores and can handle many tasks at once. **Concurrency** is the practice of running independent tasks in parallel. Tasks that don't depend on each other — like downloading 50 different map tiles, or rendering 12 independent video clips — are a perfect fit.

There are two common tools for this in Python:
- **`ThreadPoolExecutor`** — for tasks that spend most of their time *waiting* (like downloading files from the internet). Multiple threads run simultaneously, each waiting for a different server.
- **`ProcessPoolExecutor`** — for tasks that do heavy *computation* (like rendering video with FFmpeg). Multiple CPU cores each run a different task simultaneously.

### Why Does This Matter?

A typical pipeline run might download 200 tiles, taking ~10 seconds each if done one at a time. With 10 concurrent downloads, that same work takes ~1 second. For clip rendering, the gains are similar. This is likely a **3x–8x speedup** on overall pipeline time.

### A Word of Caution

Not every part of this pipeline can be parallelized. Some tile servers (like OpenStreetMap's Nominatim) explicitly limit you to 1 request per second to prevent abuse. Respecting rate limits is important — both ethically and practically, since ignoring them gets your IP blocked.

---

## 4. Error Handling

### The Concept

Right now, there are many places in the code that look like this:

```python
try:
    do_something_important()
except Exception:
    pass  # just move on
```

This is sometimes called **"swallowing errors"** or **"silent failure"**. When a render step fails silently, you might end up with a final video that is missing clip 7 and you have absolutely no idea why.

Good error handling is about being **explicit about what can go wrong**, and **choosing deliberately what to do when it does**. There are two main strategies:

1. **Fail loud** — If something critical fails (like the Gemini API refusing to respond), stop the pipeline and tell the user exactly what happened and why.
2. **Fail gracefully** — If something non-critical fails (like sourcing a YouTube clip), use a known-good fallback (like a generated map visual) and log a warning so the user knows.

**Custom exception types** are a big part of this. Instead of a generic `RuntimeError`, you raise a `GeminiAPIError` or a `TileDownloadError`. This makes error messages much more meaningful and also makes it possible to catch *specific* failures and handle them differently.

### The logging module

Python's built-in `logging` module is the professional alternative to `print()`. It supports log levels (DEBUG, INFO, WARNING, ERROR), can write to files, can be turned on/off without deleting code, and timestamps every entry. Switching from `print` to `logging` is one of the simplest things you can do that immediately makes a project feel more professional.

---

## 5. Prompt Engineering

### The Concept

The prompts sent to Gemini are the "instructions" that determine the quality of the output. Right now, they are large hardcoded strings embedded inside Python files. This has a few problems:

- **Hard to iterate on**: to try a different prompt, you have to edit Python code and re-run. A dedicated prompt file is easier to edit, version-control, and test.
- **No examples**: the prompts tell Gemini *what format to return*, but don't show it an example of a *good* output. Showing examples (called **few-shot prompting**) is one of the most effective techniques for improving LLM output consistency.
- **No validation**: Gemini sometimes returns JSON that is slightly wrong — a field missing here, a wrong type there. Currently, this causes a crash or a silent skip much later in the pipeline. Instead, the response should be validated immediately against a schema.

### Why Does This Matter?

Better prompts = better scripts = better videos. Validated outputs = fewer mysterious crashes. Iterating on prompts without touching Python code = faster improvement cycles.

### What This Looks Like in Practice

- Prompts live in their own files (e.g., `prompts/region_script.txt`), making them easy to edit and track over time.
- A "prompt registry" loads the right template based on context.
- The JSON response from Gemini is instantly parsed into a Pydantic model — the same Pydantic from #2 above — and any missing or malformed fields are caught immediately with a clear error.
- A few example "ideal" scripts are embedded into the prompts so Gemini has a target to aim for.

---

## 6. Project Cleanup

### The Concept

The repository currently has two versions of the pipeline: the original monolithic `geo_short_maker.py` (3,000+ lines) and the newer modular `pipeline/` package. Having both is confusing — which one is "the real one"? Which one gets bugfixes?

The answer is: the modular `pipeline/` package is the future. The monolith should be archived or deleted. The project should have a single, clean entry point that is immediately obvious to anyone who clones the repo.

This is called **reducing technical debt** — the accumulated "I'll fix this later" decisions that slow down development over time.

---

## Summary of Benefits

| Change | Benefit |
| :--- | :--- |
| DDD / Bounded Contexts | Easier to add features without breaking existing ones |
| Pydantic Config | Instant, readable errors for misconfiguration |
| Concurrency | 3–8x faster pipeline execution |
| Error Handling | No more silent failures; clear, actionable error messages |
| Prompt Engineering | Better videos, fewer crashes from LLM output, faster iteration |
| Project Cleanup | One clear entry point; no confusion about which file is "real" |
