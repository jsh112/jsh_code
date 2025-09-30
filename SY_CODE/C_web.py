# C_web.py
import threading, time
from flask import Flask, request, jsonify, render_template_string

_app = Flask(__name__)

# ====== 런타임 공유 상태 ======
_state = {
    "mode": "save",           # "save" | "climb"
    "selected": None,         # {"sector","level","color"}
    "records_count": 0,
    "last_record": None,      # [part/id label, id, cx, cy] 등
    "fps": 0.0,
    "stop": False,
    "reset": False,
    "rescan": False,
}
_meta_event = threading.Event()
_server_started = False

_ALLOWED_COLORS = [
    "black","blue","gray","green","lime","orange","pink","purple","red","sky","white","yellow"
]

_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>캡스톤 컨트롤 패널</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    h1 { margin: 0 0 12px 0; }
    fieldset { margin: 16px 0; padding: 16px; }
    label { margin-right: 12px; }
    .row { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
    .pill { padding:6px 10px; border:1px solid #ccc; border-radius:999px; cursor:pointer; }
    .pill.selected { background:#111; color:#fff; border-color:#111; }
    button { padding:10px 14px; margin-right:8px; cursor:pointer; border-radius:8px; border:1px solid #ccc; }
    .status { background:#fafafa; border:1px solid #eee; padding:12px; margin-top:10px; border-radius:8px; }
    .muted { color:#777; }
    .grid { display:grid; grid-template-columns: 140px 1fr; gap:8px 12px; align-items:center; }
    .danger { background:#e53935; color:#fff; border:none; }
    .warn { background:#f9a825; color:#111; border:none; }
    .ok { background:#1e88e5; color:#fff; border:none; }
    .modebtn { background:#fff; }
    .modebtn.active { background:#111; color:#fff; border-color:#111; }
    .badge { display:inline-block; padding:4px 8px; background:#eef2ff; color:#1e3a8a; border-radius:6px; }
  </style>
</head>
<body>
  <h1>캡스톤 컨트롤 패널</h1>

  <fieldset>
    <legend>현재 모드</legend>
    <div class="row">
      <button id="btnModeSave"  class="modebtn">경로 저장 모드</button>
      <button id="btnModeClimb" class="modebtn">사용자 등반 모드</button>
      <span id="modeHint" class="badge">save</span>
    </div>
    <div class="muted" style="margin-top:8px">
      경로 저장 모드는 <b>C_Save_Route.py</b>, 사용자 등반 모드는 <b>C_Climbing.py</b>가 실행합니다.
    </div>
  </fieldset>

  <form id="selectForm" method="POST" action="/select">
    <fieldset>
      <legend>메타 선택 (섹터·난이도·색상)</legend>
      <div class="grid">
        <div>난이도</div>
        <div class="row" id="levelRow">
          {% for lv in ["level1","level2","level3","level4"] %}
            <label class="pill"><input type="radio" name="level" value="{{lv}}" style="display:none">{{lv}}</label>
          {% endfor %}
        </div>

        <div>섹터</div>
        <div class="row" id="sectorRow">
          {% for s in ["sector1","sector2","sector3","sector4"] %}
            <label class="pill"><input type="radio" name="sector" value="{{s}}" style="display:none">{{s}}</label>
          {% endfor %}
        </div>

        <div>색상</div>
        <div class="row" id="colorRow">
          {% for c in colors %}
            <label class="pill"><input type="radio" name="color" value="{{c}}" style="display:none">{{c}}</label>
          {% endfor %}
          <label class="pill"><input type="radio" name="color" value="all" style="display:none">all</label>
        </div>
      </div>
      <div style="margin-top:12px">
        <button class="ok" type="submit">이 메타로 시작</button>
        <span class="muted">현재 선택된 <b>모드</b>에 따라, C_Save_Route 또는 C_Climbing에서 이 메타를 사용합니다.</span>
      </div>
    </fieldset>
  </form>

  <fieldset>
    <legend>실시간 상태 / 제어</legend>
    <div class="status" id="statusBox">
      <div><b>모드</b>: <span id="modeLabel" class="muted">-</span></div>
      <div><b>선택 메타</b>: <span id="selMeta" class="muted">-</span></div>
      <div><b>개수</b>: <span id="recCount">0</span></div>
      <div><b>마지막</b>: <span id="lastRec" class="muted">-</span></div>
      <div><b>FPS</b>: <span id="fps">0.0</span></div>
    </div>
    <div style="margin-top:10px">
      <button class="danger" id="btnStop">종료</button>
      <button class="warn"   id="btnReset">초기화</button>
      <button class="ok"     id="btnRescan">YOLO 재스캔</button>
      <div class="muted" style="margin-top:6px">* 종료/초기화/재스캔은 <b>현재 모드</b>에 적용됩니다.</div>
    </div>
  </fieldset>

<script>
  function wirePills(rowId) {
    const row = document.getElementById(rowId);
    row.querySelectorAll('.pill').forEach(p => {
      p.addEventListener('click', () => {
        row.querySelectorAll('.pill').forEach(x=>x.classList.remove('selected'));
        p.classList.add('selected');
        const input = p.querySelector('input');
        input.checked = true;
      });
    });
  }
  wirePills('levelRow'); wirePills('sectorRow'); wirePills('colorRow');

  async function getState() {
    const r = await fetch('/api/state'); return await r.json();
  }
  async function poll() {
    try {
      const j = await getState();
      document.getElementById('recCount').textContent = j.records_count ?? 0;
      document.getElementById('fps').textContent = (j.fps ?? 0).toFixed(1);
      const lr = j.last_record ? `[${j.last_record[0]}] id=${j.last_record[1]} @ (${j.last_record[2]}, ${j.last_record[3]})` : '-';
      document.getElementById('lastRec').textContent = lr;
      const sm = j.selected ? `${j.selected.sector} · ${j.selected.level} · ${j.selected.color}` : '-';
      document.getElementById('selMeta').textContent = sm;
      document.getElementById('modeLabel').textContent = j.mode;
      document.getElementById('modeHint').textContent = j.mode;
      // 버튼 하이라이트
      document.getElementById('btnModeSave').classList.toggle('active', j.mode === 'save');
      document.getElementById('btnModeClimb').classList.toggle('active', j.mode === 'climb');
    } catch (e) {}
    setTimeout(poll, 500);
  }
  poll();

  async function postJSON(url, data) {
    try {
      await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: data ? JSON.stringify(data) : null});
    } catch(e) {}
  }
  document.getElementById('btnStop').addEventListener('click', ()=>postJSON('/api/stop'));
  document.getElementById('btnReset').addEventListener('click', ()=>postJSON('/api/reset'));
  document.getElementById('btnRescan').addEventListener('click', ()=>postJSON('/api/rescan'));

  document.getElementById('btnModeSave').addEventListener('click', ()=>postJSON('/api/mode', {mode:'save'}));
  document.getElementById('btnModeClimb').addEventListener('click', ()=>postJSON('/api/mode', {mode:'climb'}));
</script>
</body></html>
"""

@_app.route("/", methods=["GET"])
def home():
    colors = _state.get("_colors") or _ALLOWED_COLORS
    return render_template_string(_HTML, colors=colors)

@_app.route("/select", methods=["POST"])
def select():
    level  = request.form.get("level", "level1")
    sector = request.form.get("sector", "sector1")
    color  = request.form.get("color", "all")
    _state["selected"] = {"level": level, "sector": sector, "color": color}
    _meta_event.set()
    return ("<script>history.back()</script>", 200)

@_app.route("/api/state", methods=["GET"])
def api_state():
    s = {
        "mode": _state["mode"],
        "selected": _state["selected"],
        "records_count": _state["records_count"],
        "last_record": _state["last_record"],
        "fps": _state["fps"],
    }
    return jsonify(s)

@_app.route("/api/stop", methods=["POST"])
def api_stop():
    _state["stop"] = True
    return ("OK", 200)

@_app.route("/api/reset", methods=["POST"])
def api_reset():
    _state["reset"] = True
    return ("OK", 200)

@_app.route("/api/rescan", methods=["POST"])
def api_rescan():
    _state["rescan"] = True
    return ("OK", 200)

@_app.route("/api/mode", methods=["POST"])
def api_mode():
    try:
        j = request.get_json(silent=True) or {}
        mode = j.get("mode", "").strip().lower()
        if mode in ("save","climb"):
            _state["mode"] = mode
            return ("OK", 200)
    except Exception:
        pass
    return ("Bad Request", 400)

def _run_server():
    _app.run(host="127.0.0.1", port=5002, debug=False, use_reloader=False)

def _ensure_server(colors_from_yolo=None):
    global _server_started
    if colors_from_yolo:
        _state["_colors"] = colors_from_yolo
    if not _server_started:
        t = threading.Thread(target=_run_server, daemon=True)
        t.start()
        _server_started = True
        time.sleep(0.2)

# ===== 외부 API =====

def choose_meta_via_web(yolo_class_names=None):
    # YOLO 클래스에서 색상 후보 추출
    colors = None
    if yolo_class_names:
        toks = set()
        for name in yolo_class_names:
            n = str(name).strip().lower()
            for sep in ("_", "-", "/", " "):
                if sep in n: n = n.split(sep)[-1]
            toks.add(n)
        colors = [c for c in _ALLOWED_COLORS if c in toks] or _ALLOWED_COLORS
    _ensure_server(colors_from_yolo=colors)
    _meta_event.clear()
    _state["selected"] = None
    _meta_event.wait()
    return _state["selected"]

def meta_to_csv_filename(meta: dict) -> str:
    return f"{meta.get('sector','sector1')}_{meta.get('level','level1')}_{meta.get('color','all')}.csv"

def update_state(records_count: int = None, last_record=None, fps: float = None, mode: str = None):
    if records_count is not None: _state["records_count"] = int(records_count)
    if last_record is not None:   _state["last_record"] = list(last_record)
    if fps is not None:           _state["fps"] = float(fps)
    if mode in ("save","climb"):  _state["mode"] = mode

def consume_flags():
    out = {"stop": _state["stop"], "reset": _state["reset"], "rescan": _state["rescan"]}
    _state["stop"] = _state["reset"] = _state["rescan"] = False
    return out
