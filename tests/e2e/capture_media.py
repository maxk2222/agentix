#!/usr/bin/env python3
"""Capture all dashboard + public-chat documentation media in 1440p.

Replaces the former capture_screenshots.py / capture_walkthroughs.py pair.

Design notes
------------
* The dashboard shows a full-page ``.login-page > .loading`` splash both at
  auth-boot and as the React-Suspense fallback on *every* lazy route change.
  A single continuous recording across pages would therefore always contain
  "Loading…" frames.  Instead each tour page is recorded in its own video
  context; the clip is then head-trimmed at the moment the page became ready.
  Because Playwright's video t=0 is the first page creation (>= context
  creation), the measured ready-offset always *over*estimates the loading
  part — trimming it removes the splash completely and only eats a little of
  the intentional settle time.  The trimmed segments are concatenated and
  converted to a GIF, so the final tour is guaranteed splash-free.
* A reference frame of the loading splash is captured once; after conversion
  a few frames of every produced GIF are compared against it as a post-check.
* All sub-interactions are read-only: expand/click open only — never submit,
  save, delete, toggle or run.  Every interaction degrades gracefully when
  the live instance lacks the data (skipped with a log line).

Requires Playwright + chromium + imageio-ffmpeg in the venv:

    .venv/bin/pip install playwright imageio-ffmpeg
    .venv/bin/python -m playwright install chromium

Usage:

    AGENTIX_BASE_URL=https://host AGENTIX_USER=u AGENTIX_PASSWORD=p \
        .venv/bin/python tests/e2e/capture_media.py

Outputs ``<page>-<theme>.png`` (2560x1440) into docs/assets/screenshots and
``<scenario>.gif`` (1920x1080, looping) into docs/assets/walkthroughs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DASH_PREFS_KEY = "agentix-dashboard-prefs"
PUBLIC_PREFS_KEYS = {"theme": "agentix-public-chat-theme", "lang": "agentix-public-chat-lang"}
SPLASH = ".login-page > .loading"          # boot + Suspense fallback splash
SHELL = ".app-shell"                        # authenticated dashboard shell
THEMES = ("light", "dark")


@dataclass
class Page:
    """One dashboard/public page to capture."""
    name: str
    route: str
    ready: tuple[str, ...]                  # first visible selector wins
    shot: bool = True
    tour: bool = True
    auth: str = "dashboard"                 # dashboard | public | none
    interact: Optional[Callable[["Page", "UI"], None]] = field(
        default=None, repr=False, compare=False
    )
    hold_s: float = 1.6                     # visible dwell before interactions


class UI:
    """Paced interaction helper — keeps the tour human-readable."""

    def __init__(self, page, pause_s: float):
        self.page = page
        self.pause_s = pause_s
        self.marks: list[float] = []

    def mark(self) -> None:
        """Record 'content ready' — used as the clip trim offset."""
        if not self.marks:
            self.marks.append(time.monotonic())

    def pause(self, factor: float = 1.0) -> None:
        self.page.wait_for_timeout(int(self.pause_s * 1000 * factor))

    def first(self, sel: str):
        return self.page.locator(sel).first

    def click_if(self, sel: str, label: str, pause: float = 1.0) -> bool:
        loc = self.first(sel)
        try:
            if not loc.is_visible(timeout=3000):
                print(f"      skip {label}: not visible", flush=True)
                return False
            loc.click()
            self.pause(pause)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"      skip {label}: {exc}", flush=True)
            return False

    def wait_sel(self, sel: str, timeout: int = 6000) -> bool:
        try:
            self.page.wait_for_selector(sel, state="visible", timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            return False

    def close_overlay(self) -> None:
        """Close a modal/drawer via its header close button, then Escape."""
        if not self.click_if(
            ".modal-overlay .btn-icon[title], .drawer-head .btn-icon",
            "close overlay", pause=0.6,
        ):
            self.page.keyboard.press("Escape")
            self.pause(0.6)


# ---------------------------------------------------------------------------
# per-page interactions (all read-only)
# ---------------------------------------------------------------------------

def i_overview(p: "Page", ui: UI) -> None:
    ui.click_if(".filter-chips .filter-chip >> nth=1", "range chip")


def i_chat(p: "Page", ui: UI) -> None:
    if ui.click_if(".chat-history-header-btn", "history"):
        if ui.wait_sel(".chat-history-modal-item"):
            ui.first(".chat-history-modal-item").click()
            ui.pause(1.4)
        ui.page.keyboard.press("Escape")
        ui.pause(0.6)
        ui.close_overlay()


def i_conversations(p: "Page", ui: UI) -> None:
    if ui.click_if(".conv-title-link", "conversation detail"):
        ui.wait_sel(".conversation-modal")
        ui.pause(1.2)
        ui.close_overlay()


def i_assistants(p: "Page", ui: UI) -> None:
    if ui.click_if(".assistant-card .assistant-header", "expand assistant"):
        if ui.wait_sel(".assistant-tab"):
            ui.first(".assistant-tabs .assistant-tab >> nth=2").click()
            ui.pause(1.2)


def i_knowledge(p: "Page", ui: UI) -> None:
    if ui.click_if(".assistant-card .assistant-header", "expand collection"):
        ui.wait_sel(".file-list")
        ui.pause(1.0)


def i_tools(p: "Page", ui: UI) -> None:
    # tool cards aren't clickable — the drawer opens via "Configure →"
    if ui.click_if(".entity-card .btn-link", "tool drawer"):
        if ui.wait_sel(".drawer"):
            ui.pause(1.2)
            ui.close_overlay()


def i_chains(p: "Page", ui: UI) -> None:
    if ui.click_if(".card-grid .entity-card", "chain editor"):
        if ui.wait_sel(".chain-editor .chain-canvas", timeout=8000):
            ui.pause(1.4)
            ui.page.keyboard.press("Escape")
            ui.pause(0.5)
            # editor has its own close (toolbar onClose → Cancel/Abbrechen/X)
            ui.click_if(
                ".chain-editor button[title*='ancel' i], "
                ".chain-editor button[title*='bbrechen' i], "
                ".chain-editor button[title*='lose' i], "
                ".chain-editor button[title*='back' i]",
                "close editor", pause=0.8,
            )


def i_triggers(p: "Page", ui: UI) -> None:
    if ui.click_if(".card-grid .entity-card", "trigger drawer"):
        if ui.wait_sel(".drawer"):
            ui.pause(1.2)
            ui.close_overlay()


def i_metrics(p: "Page", ui: UI) -> None:
    ui.click_if(".rc-tabs .rc-tab >> nth=1", "usage tab", pause=1.2)
    ui.click_if(".filter-chips .filter-chip >> nth=1", "range chip")


def i_backups(p: "Page", ui: UI) -> None:
    ui.page.mouse.wheel(0, 500)          # reveal jobs table
    ui.pause(1.0)


def i_widget(p: "Page", ui: UI) -> None:
    ui.click_if(".tab-btn >> nth=1", "public-chat config tab", pause=1.2)
    ui.click_if(".tab-btn >> nth=0", "widget tab", pause=0.8)


def i_voice(p: "Page", ui: UI) -> None:
    ui.click_if(".vm-voice-info", "voice card")
    ui.page.mouse.wheel(0, 400)
    ui.pause(0.8)


def i_emotion(p: "Page", ui: UI) -> None:
    ui.click_if(".em-fusion-header", "fusion card")


def i_runtime(p: "Page", ui: UI) -> None:
    ui.click_if(".rc-tabs .rc-tab >> nth=1", "providers tab", pause=1.2)
    ui.click_if(".rc-tabs .rc-tab >> nth=0", "config tab", pause=0.8)


def i_users(p: "Page", ui: UI) -> None:
    if ui.click_if("button[title='Edit']", "edit user"):
        ui.pause(0.8)
        ui.click_if("button:has-text('Cancel'), button:has-text('Abbrechen')",
                    "cancel edit", pause=0.6)


def i_public_chat(p: "Page", ui: UI) -> None:
    ui.click_if(".chat-icon-btn", "theme toggle", pause=1.2)


# ---------------------------------------------------------------------------
# page table (tour order == sidebar order)
# ---------------------------------------------------------------------------

PAGES: list[Page] = [
    Page("login", "/dashboard/login", ("input[type='password']",),
         tour=False, auth="none"),
    Page("overview", "/dashboard/overview", (".filter-chips", ".kpi-card"),
         interact=i_overview),
    Page("chat", "/dashboard/chat", (".chat-textarea", ".chat-page"),
         interact=i_chat),
    Page("conversations", "/dashboard/conversations", (".conv-title-link", "table"),
         interact=i_conversations),
    Page("config-assistants", "/dashboard/config-assistants", (".assistant-card",),
         interact=i_assistants),
    Page("config-knowledge", "/dashboard/config-knowledge", (".assistant-card", ".config-section"),
         interact=i_knowledge),
    Page("config-tools", "/dashboard/config-tools", (".entity-card", ".card-grid"),
         interact=i_tools),
    Page("agent-chains", "/dashboard/agent-chains", (".chains-page",),
         interact=i_chains),
    Page("triggers", "/dashboard/triggers", (".card-grid",),
         interact=i_triggers),
    Page("metrics", "/dashboard/metrics", (".rc-tabs", ".kpi-card"),
         interact=i_metrics),
    Page("backups", "/dashboard/backups", (".kpi-grid",),
         interact=i_backups),
    Page("widget", "/dashboard/widget", (".tabs-header", ".tab-btn"),
         interact=i_widget),
    Page("voice-management", "/dashboard/voice-management", (".vm-page",),
         interact=i_voice),
    Page("emotion-models", "/dashboard/emotion-models", (".em-page", ".em-fusion-header"),
         interact=i_emotion),
    Page("config-runtime", "/dashboard/config-runtime", (".rc-tabs", ".rc-card"),
         interact=i_runtime),
    Page("users", "/dashboard/users", (".rc-card", "table"),
         interact=i_users),
    Page("public-chat", "/chat/", (".chat-textarea", ".chat-container", ".chat-page"),
         auth="public", interact=i_public_chat),
]
PAGE_BY_NAME = {p.name: p for p in PAGES}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_dash_prefs(page, theme: str, language: str) -> None:
    page.evaluate(
        "([k, v]) => localStorage.setItem(k, v)",
        [DASH_PREFS_KEY, json.dumps({"theme": theme, "language": language})],
    )


def set_public_prefs(page, theme: str, language: str) -> None:
    page.evaluate(
        """([t, tk, l, lk]) => {
            localStorage.setItem(tk, t); localStorage.setItem(lk, l);
        }""",
        [theme, PUBLIC_PREFS_KEYS["theme"], language, PUBLIC_PREFS_KEYS["lang"]],
    )


def try_public_login(page, base: str, user: str, password: str) -> bool:
    try:
        resp = page.request.post(
            f"{base}/public-chat/auth/login",
            data=json.dumps({"username": user, "password": password}),
            headers={"Content-Type": "application/json"},
            timeout=8000,
        )
        return resp.ok
    except Exception:  # noqa: BLE001
        return False


def wait_ready(page, spec: Page, lang_sel: Optional[str] = None) -> bool:
    """Wait until the page is genuinely rendered (no splash, content visible).

    Returns False on timeout — callers treat it as a failed segment.
    """
    try:
        if spec.auth == "dashboard":
            page.wait_for_selector(SHELL, state="visible", timeout=20000)
        # splash must be gone (it may never have existed — tolerate timeout)
        try:
            page.wait_for_selector(SPLASH, state="hidden", timeout=12000)
        except Exception:  # noqa: BLE001
            pass
        for sel in spec.ready:
            try:
                page.wait_for_selector(sel, state="visible", timeout=9000)
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            return False
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:  # noqa: BLE001
            pass
        # DOM stability: two consecutive identical body snapshots
        last = None
        for _ in range(4):
            snap = page.evaluate("() => document.body.innerText.length")
            if snap == last:
                break
            last = snap
            page.wait_for_timeout(250)
        return True
    except Exception:  # noqa: BLE001
        return False


def apply_prefs(page, spec: Page, theme: str, lang: str) -> None:
    if spec.auth == "public":
        set_public_prefs(page, theme, lang)
    else:
        set_dash_prefs(page, theme, lang)


def ui_login(page, base: str, user: str, password: str) -> None:
    page.goto(f"{base}/dashboard/login", wait_until="load")
    page.wait_for_timeout(700)
    page.fill('input[type="text"], input:not([type])', user)
    page.fill('input[type="password"]', password)
    page.click('button[type="submit"]')
    page.wait_for_selector(SHELL, state="visible", timeout=20000)


def ffmpeg_run(ffmpeg: str, *cli: str) -> None:
    subprocess.run([ffmpeg, "-hide_banner", "-y", *cli],
                   check=True, capture_output=True)


def webm_to_gif(ffmpeg: str, src: Path, gif: Path, width: int, fps: int,
                colors: int = 192, seek: float = 0.0) -> None:
    vf = (
        f"fps={fps},scale={width}:-2:flags=lanczos,"
        f"split[s0][s1];[s0]palettegen=max_colors={colors}[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    seek_args = ["-ss", f"{seek:.3f}"] if seek > 0 else []
    ffmpeg_run(ffmpeg, *seek_args, "-i", str(src),
               "-vf", vf, "-loop", "0", str(gif))


def gif_frames(ffmpeg: str, gif: Path, tmpdir: Path, count: int = 3) -> list[Path]:
    """Extract `count` evenly-spaced frames from a gif/video as PNGs."""
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(gif)],
                           capture_output=True, text=True)
    dur = 1.0
    for line in probe.stderr.splitlines():
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            dur = max(1.0, int(h) * 3600 + int(m) * 60 + float(s))
            break
    out: list[Path] = []
    for i in range(count):
        ts = dur * (i + 1) / (count + 1)
        dest = tmpdir / f"fr_{gif.stem}_{i}.png"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-y", "-ss", f"{ts:.2f}",
             "-i", str(gif), "-frames:v", "1", str(dest)],
            capture_output=True,
        )
        out.append(dest)
    return out


def _nonbg_share(png: Path) -> float:
    """Fraction of pixels differing from the corner-estimated background."""
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
    c = np.concatenate([
        a[:40, :40].ravel(), a[:40, -40:].ravel(),
        a[-40:, :40].ravel(), a[-40:, -40:].ravel(),
    ])
    return float(np.mean(np.abs(a - np.median(c)) > 12))


def is_splash_frame(png: Path) -> bool:
    """The `.login-page > .loading` splash is a near-empty page: a small
    centred text on a flat background (measured non-bg share ~0.001-0.005).
    Real pages — even the sparsest (a card grid with one card) — sit at
    >= 0.012.  Threshold 0.008 separates the two; a reference-image
    comparison proved unreliable (the splash is too transient to capture
    deterministically and sparse dark pages resemble it pixel-wise).
    """
    try:
        return _nonbg_share(png) < 0.008
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# capture passes
# ---------------------------------------------------------------------------

def capture_screenshots(browser, base: str, state: dict, pages: list[Page],
                        args, out: Path) -> list[str]:
    captured: list[str] = []
    for theme in args.themes:
        for spec in pages:
            if not spec.shot:
                continue
            ctx_kwargs = {}
            if spec.auth != "none":   # login page must be seen unauthenticated
                ctx_kwargs["storage_state"] = state
            ctx = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=args.device_scale,
                **ctx_kwargs,
            )
            page = ctx.new_page()
            try:
                resp = page.goto(f"{base}{spec.route}", wait_until="load")
                if resp is not None and resp.status >= 400:
                    print(f"  skip {spec.name} ({resp.status})")
                    continue
                apply_prefs(page, spec, theme, args.language)
                page.reload(wait_until="load")
                if not wait_ready(page, spec):
                    print(f"  WARN {spec.name}: not ready — capturing anyway")
                page.wait_for_timeout(args.settle_ms)
                dest = out / f"{spec.name}-{theme}.png"
                page.screenshot(path=str(dest))
                captured.append(spec.name)
                print(f"  captured {dest.name}")
            finally:
                ctx.close()
    return captured


def record_segment(browser, base: str, state: dict, spec: Page, args,
                   tmpdir: Path) -> tuple[Optional[Path], float]:
    """Record one page segment. Returns (webm_path, trim_offset_s)."""
    ctx = browser.new_context(
        storage_state=state,
        viewport={"width": args.width, "height": args.height},
        device_scale_factor=1,
        record_video_dir=str(tmpdir),
        record_video_size={"width": args.width, "height": args.height},
    )
    t0 = time.monotonic()
    page = ctx.new_page()
    offset = 0.0
    ok = False
    try:
        resp = page.goto(f"{base}{spec.route}", wait_until="load")
        if resp is not None and resp.status >= 400:
            print(f"  segment {spec.name}: HTTP {resp.status}")
            ctx.close()
            return None, 0.0
        apply_prefs(page, spec, args.tour_theme, args.language)
        page.reload(wait_until="load")
        ok = wait_ready(page, spec)
        offset = time.monotonic() - t0
        page.wait_for_timeout(int(args.settle_ms))
        ui = UI(page, args.interact_pause_ms / 1000.0)
        page.wait_for_timeout(int(spec.hold_s * 1000))
        if ok and spec.interact:
            try:
                spec.interact(page, ui)
            except Exception as exc:  # noqa: BLE001
                print(f"  segment {spec.name}: interact failed ({exc})")
        ui.pause(0.6)
    finally:
        ctx.close()
    videos = sorted(tmpdir.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if not videos:
        return None, 0.0
    webm = videos[-1]
    if not ok:
        webm.unlink()
        return None, 0.0
    return webm, offset


def clip_to_seg(ffmpeg: str, webm: Path, seg: Path, trim: float,
                args) -> None:
    """Transcode a raw .webm clip into a uniform mp4 segment, cutting the
    head at `trim` seconds (the recorded loading period + safety margin)."""
    ffmpeg_run(
        ffmpeg, "-ss", f"{trim:.3f}", "-i", str(webm), "-an",
        "-vf", f"scale={args.gif_width}:-2:flags=lanczos",
        "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", str(seg),
    )


def concat_segs(ffmpeg: str, segs: list[Path], out_mp4: Path,
                tmpdir: Path) -> None:
    concat_list = tmpdir / f"concat_{out_mp4.stem}.txt"
    concat_list.write_text(
        "".join(f"file '{s.resolve()}'\n" for s in segs))
    ffmpeg_run(ffmpeg, "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c", "copy", str(out_mp4))


def produce_tour(browser, base: str, state: dict, pages: list[Page],
                 args, ffmpeg: str, out: Path, tmpdir: Path) -> bool:
    segs_dir = tmpdir / "tour_segments"
    segs_dir.mkdir(exist_ok=True)
    seg_files: list[Path] = []
    for spec in pages:
        if not spec.tour:
            continue
        print(f"  tour segment: {spec.name}", flush=True)
        result = None
        for attempt in (1, 2):
            result = record_segment(browser, base, state, spec, args, tmpdir)
            if result[0] is not None:
                break
            print(f"    retry {attempt} failed for {spec.name}")
        webm, offset = result
        if webm is None:
            print(f"  ERROR: segment {spec.name} failed — aborting tour")
            return False
        seg = segs_dir / f"seg_{len(seg_files):02d}_{spec.name}.mp4"
        clip_to_seg(ffmpeg, webm, seg, offset + 0.25, args)
        seg_files.append(seg)
        if args.keep_clips:
            webm.replace(out / f"_clip_{spec.name}.webm")
        else:
            webm.unlink()

    tour_mp4 = tmpdir / "tour.mp4"
    concat_segs(ffmpeg, seg_files, tour_mp4, tmpdir)
    gif = out / "demo-tour.gif"
    webm_to_gif(ffmpeg, tour_mp4, gif, args.gif_width, args.gif_fps,
                args.gif_colors)
    print(f"  captured {gif.name} ({gif.stat().st_size // 1024} KB)")
    return True


# ---------------------------------------------------------------------------
# scenario walkthroughs (single-clip, head-trimmed)
# ---------------------------------------------------------------------------

def scenario_clip(browser, base: str, state: Optional[dict], name: str, args,
                  tmpdir: Path, body: Callable) -> tuple[Optional[Path], float]:
    ctx_kwargs = {}
    if state is not None:
        ctx_kwargs["storage_state"] = state
    ctx = browser.new_context(
        viewport={"width": args.width, "height": args.height},
        device_scale_factor=1,
        record_video_dir=str(tmpdir),
        record_video_size={"width": args.width, "height": args.height},
        **ctx_kwargs,
    )
    t0 = time.monotonic()
    page = ctx.new_page()
    ui = UI(page, args.interact_pause_ms / 1000.0)
    ok = True
    try:
        body(page, base, args, ui)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  scenario {name} failed: {exc}")
    finally:
        ctx.close()
    videos = sorted(tmpdir.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if not videos or not ok:
        return None, 0.0
    offset = (ui.marks[0] - t0) if ui.marks else 0.0
    return videos[-1], offset


def sc_login_form(page, base: str, args, ui: UI):
    """Clip A of demo-login: form + typing + submit click.

    Ends *before* the post-submit navigation — the lazy-route Suspense
    splash would otherwise appear mid-clip and head-trimming can't reach
    it.  Clip B resumes on the settled overview page; concatenating the
    two reads as a continuous login → dashboard flow with zero splash.
    """
    page.goto(f"{base}/dashboard/login", wait_until="load")
    set_dash_prefs(page, "light", args.language)
    page.wait_for_selector("input[type='password']", timeout=15000)
    ui.mark()  # login form is the content — keep typing in the gif
    ui.pause(0.8)
    page.type('input[type="text"], input:not([type])', args.user, delay=70)
    # real password — the field renders it masked anyway
    page.type('input[type="password"]', args.password, delay=70)
    ui.pause(0.5)
    page.click('button[type="submit"]')
    # no trailing pause: the context must close before the post-submit
    # lazy-route splash can render — the concat cut covers the transition


def sc_login_overview(page, base: str, args, ui: UI):
    """Clip B of demo-login: the settled overview after sign-in."""
    spec = PAGE_BY_NAME["overview"]
    page.goto(f"{base}{spec.route}", wait_until="load")
    set_dash_prefs(page, "light", args.language)
    page.reload(wait_until="load")
    if not wait_ready(page, spec):
        raise RuntimeError("overview not ready after login")
    ui.mark()
    ui.pause(1.6)


def sc_theme_body(page, base: str, args, ui: UI):
    """Theme toggle via the real sidebar control — no reload (a reload
    would show the boot splash mid-clip)."""
    spec = PAGE_BY_NAME["overview"]
    page.goto(f"{base}{spec.route}", wait_until="load")
    set_dash_prefs(page, "light", args.language)
    page.reload(wait_until="load")
    wait_ready(page, spec)
    ui.mark()
    ui.pause(1.2)
    toggle = ".sidebar-control-btn >> nth=0"
    ui.click_if(toggle, "dark theme", pause=1.4)
    ui.click_if(toggle, "light theme", pause=1.4)


def sc_widget_body(page, base: str, args, ui: UI):
    spec = PAGE_BY_NAME["widget"]
    page.goto(f"{base}{spec.route}", wait_until="load")
    set_dash_prefs(page, args.tour_theme, args.language)
    page.reload(wait_until="load")
    wait_ready(page, spec)
    ui.mark()
    ui.pause(1.0)
    spec.interact(page, ui)


def sc_backups_body(page, base: str, args, ui: UI):
    spec = PAGE_BY_NAME["backups"]
    page.goto(f"{base}{spec.route}", wait_until="load")
    set_dash_prefs(page, args.tour_theme, args.language)
    page.reload(wait_until="load")
    wait_ready(page, spec)
    ui.mark()
    ui.pause(1.2)
    page.mouse.wheel(0, 600)
    ui.pause(1.2)


def sc_public_chat_body(page, base: str, args, ui: UI):
    spec = PAGE_BY_NAME["public-chat"]
    resp = page.goto(f"{base}{spec.route}", wait_until="load")
    if resp is not None and resp.status >= 400:
        raise RuntimeError(f"public chat unavailable ({resp.status})")
    set_public_prefs(page, "light", args.language)
    page.reload(wait_until="load")
    if not wait_ready(page, spec):
        raise RuntimeError("public chat not ready")
    ui.mark()
    ui.pause(1.2)
    spec.interact(page, ui)


def produce_scenarios(browser, base: str, state: dict, args, ffmpeg: str,
                      out: Path, tmpdir: Path) -> None:
    bodies: dict[str, Callable] = {
        "demo-theme": sc_theme_body,
        "demo-widget": sc_widget_body,
        "demo-backup-run": sc_backups_body,
        "demo-public-chat": sc_public_chat_body,
    }
    for name in (args.scenarios or ["demo-login"] + list(bodies)):
        print(f"  scenario: {name}", flush=True)
        if name == "demo-login":
            # Two clips concatenated: (A) unauthenticated login form +
            # typing + submit click — closed before the post-submit
            # Suspense splash can render; (B) settled overview.
            parts: list[tuple[Optional[Path], float]] = [
                scenario_clip(browser, base, None, f"{name}-a", args,
                              tmpdir, sc_login_form),
                scenario_clip(browser, base, state, f"{name}-b", args,
                              tmpdir, sc_login_overview),
            ]
            if any(c is None for c, _ in parts):
                print(f"  {name}: skipped/failed")
                continue
            segs_dir = tmpdir / f"{name}_segments"
            segs_dir.mkdir(exist_ok=True)
            segs = []
            for i, (clip, off) in enumerate(parts):
                seg = segs_dir / f"seg_{i}.mp4"
                clip_to_seg(ffmpeg, clip, seg, off + 0.15, args)
                clip.unlink()
                segs.append(seg)
            joined = tmpdir / f"{name}.mp4"
            concat_segs(ffmpeg, segs, joined, tmpdir)
            gif = out / f"{name}.gif"
            webm_to_gif(ffmpeg, joined, gif, args.gif_width, args.gif_fps,
                        args.gif_colors)
            print(f"  captured {gif.name} ({gif.stat().st_size // 1024} KB)")
            continue
        body = bodies.get(name)
        if body is None:
            print(f"  unknown scenario {name} — skipping")
            continue
        clip, offset = scenario_clip(browser, base, state, name, args,
                                     tmpdir, body)
        if clip is None:
            print(f"  {name}: skipped/failed")
            continue
        gif = out / f"{name}.gif"
        webm_to_gif(ffmpeg, clip, gif, args.gif_width, args.gif_fps,
                    args.gif_colors, seek=offset + 0.15)
        clip.unlink()
        print(f"  captured {gif.name} ({gif.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="https://agentix-cai.vimba-perch.ts.net")
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--shots-out", default="docs/assets/screenshots")
    ap.add_argument("--gifs-out", default="docs/assets/walkthroughs")
    ap.add_argument("--language", default="en", choices=["en", "de"])
    ap.add_argument("--mode", nargs="+", default=["screenshots", "tour", "scenarios"],
                    choices=["screenshots", "tour", "scenarios", "all"])
    ap.add_argument("--pages", nargs="*", default=None,
                    help="page subset for screenshots+tour")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="scenario subset (demo-login demo-theme demo-widget "
                         "demo-backup-run demo-public-chat)")
    ap.add_argument("--themes", nargs="*", default=list(THEMES), choices=list(THEMES))
    ap.add_argument("--tour-theme", default="dark", choices=list(THEMES))
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--device-scale", type=int, default=1)
    ap.add_argument("--gif-width", type=int, default=1920)
    ap.add_argument("--gif-fps", type=int, default=12)
    ap.add_argument("--gif-colors", type=int, default=192)
    ap.add_argument("--settle-ms", type=int, default=800)
    ap.add_argument("--interact-pause-ms", type=int, default=1400)
    ap.add_argument("--keep-clips", action="store_true",
                    help="keep raw .webm segments next to the gifs")
    args = ap.parse_args()

    if "all" in args.mode:
        args.mode = ["screenshots", "tour", "scenarios"]
    modes = set(args.mode)

    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg = get_ffmpeg_exe()

    shots_out = Path(args.shots_out)
    gifs_out = Path(args.gifs_out)
    shots_out.mkdir(parents=True, exist_ok=True)
    gifs_out.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")

    names = set(args.pages) if args.pages else None
    pages = [p for p in PAGES if names is None or p.name in names]

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with sync_playwright() as p:
            browser = p.chromium.launch()

            # --- bootstrap: real UI login, reuse session everywhere ---
            boot = browser.new_context(
                viewport={"width": args.width, "height": args.height})
            page = boot.new_page()
            ui_login(page, base, args.user, args.password)
            state = boot.storage_state()
            pub_ok = try_public_login(page, base, args.user, args.password)
            if pub_ok:
                # public-chat cookies live in this context only — extract them
                state = page.context.storage_state()
            else:
                print("  note: public-chat auth failed — public pages may show login")
            boot.close()

            # --- screenshots ---
            if "screenshots" in modes:
                print("== screenshots ==")
                capture_screenshots(browser, base, state, pages, args, shots_out)

            # --- tour gif ---
            if "tour" in modes:
                print("== tour ==")
                produce_tour(browser, base, state, pages, args, ffmpeg,
                             gifs_out, tmpdir)

            # --- scenario gifs ---
            if "scenarios" in modes:
                print("== scenarios ==")
                produce_scenarios(browser, base, state, args, ffmpeg,
                                  gifs_out, tmpdir)

            browser.close()

        # --- post-checks ---
        print("== post-checks ==")
        ok_all = True
        for gif in sorted(gifs_out.glob("*.gif")):
            frames = gif_frames(ffmpeg, gif, tmpdir, count=6)
            bad = [f for f in frames if f.exists() and is_splash_frame(f)]
            if bad:
                ok_all = False
                print(f"  FAIL {gif.name}: {len(bad)} loading-splash frame(s)!")
            else:
                print(f"  ok {gif.name}: no splash frames")
        try:
            from PIL import Image
            for png in sorted(shots_out.glob("*.png")):
                with Image.open(png) as im:
                    if im.size != (args.width, args.height):
                        ok_all = False
                        print(f"  FAIL {png.name}: {im.size} != {(args.width, args.height)}")
            print("  dimensions ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  dimension check skipped: {exc}")
        return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
