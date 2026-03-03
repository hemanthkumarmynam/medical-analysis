import os, time, base64, json
from datetime import datetime
import streamlit as st
import numpy as np
import cv2
from dotenv import load_dotenv
load_dotenv()

# ══ CONFIG ════════════════════════════════════════════════════════════════════
LANGS = {"English":"en","Tamil":"ta","Hindi":"hi","Telugu":"te","Malayalam":"ml",
         "Kannada":"kn","French":"fr","Spanish":"es","Arabic":"ar","German":"de","Chinese":"zh"}

UI = {
    "en": dict(title="MediVision AI",sub="Medical Image Analysis",upload="📤 Upload & Analyze",
               chat="💬 AI Chatbot",history="📋 History",how="📖 How It Works",
               analyzing="🔬 Analyzing image...",report="AI Visual Report",
               done="✅ Report ready! Go to 💬 Chatbot for questions.",
               disc="⚠️ Educational use only. Not a substitute for medical advice.",
               ph="Ask about findings, symptoms, treatments, or find a doctor...",
               find="🏥 Find Nearby Doctors",send="Send",clear="🗑️ Clear Chat",
               loc="Your city (for doctor search)",thinking="Thinking..."),
    "ta": dict(title="மெடிவிஷன் AI",sub="மருத்துவ படங்கள் பகுப்பாய்வு",
               upload="📤 பதிவேற்று",chat="💬 AI உரையாடல்",history="📋 வரலாறு",
               how="📖 எவ்வாறு",analyzing="🔬 பகுப்பாய்கிறது...",report="AI காட்சி அறிக்கை",
               done="✅ அறிக்கை தயார்!",disc="⚠️ கல்வி நோக்கங்களுக்கு மட்டுமே.",
               ph="அறிகுறிகள், சிகிச்சை அல்லது மருத்துவரை கேளுங்கள்...",
               find="🏥 அருகிலுள்ள மருத்துவர்",send="அனுப்பு",clear="🗑️ அழி",
               loc="உங்கள் நகரம்",thinking="யோசிக்கிறது..."),
    "hi": dict(title="मेडीविज़न AI",sub="चिकित्सा छवि विश्लेषण",
               upload="📤 अपलोड",chat="💬 AI चैट",history="📋 इतिहास",
               how="📖 कैसे काम करता है",analyzing="🔬 विश्लेषण हो रहा है...",
               report="AI दृश्य रिपोर्ट",done="✅ रिपोर्ट तैयार!",
               disc="⚠️ केवल शैक्षिक उपयोग के लिए।",
               ph="लक्षण, उपचार या नज़दीकी डॉक्टर पूछें...",
               find="🏥 नज़दीकी डॉक्टर",send="भेजें",clear="🗑️ साफ़ करें",
               loc="आपका शहर",thinking="सोच रहा है..."),
}
for c in ["te","ml","kn","fr","es","ar","de","zh"]: UI[c] = UI["en"]

def T(k): return UI.get(st.session_state.get("lc","en"), UI["en"]).get(k, k)

# ══ PAGE CONFIG ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="MediVision AI", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0b0f1a;color:#e2e8f0;}
[data-testid="stSidebar"]{background:#0d1120!important;border-right:1px solid #1e2d45;}
.card{background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:.9rem;}
.ca{border-left:3px solid #38bdf8;}
.sl{font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#38bdf8;margin-bottom:.4rem;}
.ht{font-size:1.9rem;font-weight:700;color:#f0f9ff;margin:0 0 .3rem;}
.hs{color:#64748b;font-size:.9rem;}
.cu{background:#1e3a5f;border-radius:14px 14px 4px 14px;padding:.7rem 1rem;margin:.4rem 0 .4rem 2rem;color:#e2e8f0;font-size:.9rem;line-height:1.6;}
.ca2{background:#111827;border:1px solid #1e2d45;border-radius:14px 14px 14px 4px;padding:.7rem 1rem;margin:.4rem 2rem .4rem 0;color:#cbd5e1;font-size:.9rem;line-height:1.7;}
.lu{text-align:right;font-size:.68rem;color:#38bdf8;font-weight:600;}
.la{font-size:.68rem;color:#94a3b8;font-weight:600;}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:999px;font-size:.7rem;font-weight:600;}
.bg{background:#052e16;color:#4ade80;border:1px solid #166534;}
.bo{background:#1c0f00;color:#fb923c;border:1px solid #9a3412;}
.br{background:#1e293b;color:#94a3b8;border:1px solid #334155;}
.disc{background:#1a0e00;border:1px solid #92400e;border-radius:10px;padding:.8rem 1rem;color:#fbbf24;font-size:.82rem;line-height:1.5;}
.hcard{background:#0f1730;border:1px solid #1e3a5f;border-radius:10px;padding:.8rem 1rem;margin:.4rem 0;cursor:pointer;}
.hcard:hover{border-color:#38bdf8;}
.dlink a{color:#38bdf8!important;text-decoration:none;}
.dlink a:hover{text-decoration:underline;}
hr{border:none;border-top:1px solid #1e2d45;margin:1rem 0;}
[data-testid="stFileUploader"]{background:#0f1e30;border:1.5px dashed #1e3a5f;border-radius:12px;padding:.4rem;}
.stButton>button{background:#0ea5e9;color:#0b0f1a;border:none;border-radius:8px;font-weight:600;padding:.45rem 1.1rem;transition:background .2s;}
.stButton>button:hover{background:#38bdf8;}
.stTextInput>div>div>input,.stTextArea textarea{background:#0f1e30!important;color:#e2e8f0!important;border:1px solid #1e3a5f!important;border-radius:8px!important;}
</style>""", unsafe_allow_html=True)

# ══ IMAGE UTILS ═══════════════════════════════════════════════════════════════
def load_img(f):
    f.seek(0); data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enh  = cv2.createCLAHE(2.5,(8,8)).apply(gray)
    _,t  = cv2.threshold(cv2.GaussianBlur(enh,(15,15),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k    = np.ones((5,5),np.uint8)
    m    = cv2.morphologyEx(t,cv2.MORPH_CLOSE,k,iterations=2)
    return cv2.morphologyEx(m,cv2.MORPH_OPEN,k,iterations=1)

def make_overlay(img, mask, color, alpha):
    ov = img.copy(); ov[mask>0] = color
    return cv2.addWeighted(img,1-alpha,ov,alpha,0)

def b64(img):
    _,buf = cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,92])
    return base64.b64encode(buf).decode()

# ══ AI VISION ═════════════════════════════════════════════════════════════════
VISION_PROMPT = """You are an expert radiologist giving an educational visual report.{lang}
Analyze this medical image carefully. Write a structured report based ONLY on what you visually see.

**1. Image Type & View** — modality, body part, orientation
**2. Visible Structures** — all anatomical structures identified
**3. Key Visual Findings** — densities, symmetry, abnormalities, notable features  
**4. Impression** — overall educational assessment
**5. Possible Conditions** *(Educational Only)* — specific to what you see
**6. Suggested Next Steps** — appropriate clinical follow-up
**7. Notice** — state this is educational only; patient must consult a qualified doctor

File: {fname}. Be specific — describe what you actually see."""

def vision_report(img, fname):
    lang = st.session_state.get("lc","en")
    lang_note = f"\nRespond entirely in {lang} language." if lang != "en" else ""
    prompt = VISION_PROMPT.format(lang=lang_note, fname=fname)
    img_b64 = b64(img)

    # Try Groq Vision
    key = os.getenv("GROQ_API_KEY")
    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            for model in ["meta-llama/llama-4-scout-17b-16e-instruct",
                          "meta-llama/llama-4-maverick-17b-128e-instruct",
                          "llama-3.2-90b-vision-preview","llama-3.2-11b-vision-preview"]:
                try:
                    r = client.chat.completions.create(
                        model=model, max_tokens=1500, temperature=0.3,
                        messages=[{"role":"user","content":[
                            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},
                            {"type":"text","text":prompt}]}])
                    txt = r.choices[0].message.content.strip()
                    if len(txt) > 150: return txt, "Groq Vision"
                except: continue
        except: pass

    # Try Gemini Vision
    key = os.getenv("GEMINI_API_KEY")
    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            mn = st.session_state.get("_gm")
            if not mn:
                try:
                    for m in genai.list_models():
                        if "generateContent" in getattr(m,"supported_generation_methods",[]) \
                           and ("flash" in m.name or "pro" in m.name):
                            mn = m.name; break
                except: pass
                st.session_state["_gm"] = mn or "models/gemini-1.5-flash"
                mn = st.session_state["_gm"]
            for _ in range(2):
                try:
                    r = genai.GenerativeModel(mn).generate_content(
                        [{"inline_data":{"mime_type":"image/jpeg","data":img_b64}}, prompt])
                    txt = r.text.strip()
                    if len(txt) > 150: return txt, "Gemini Vision"
                except Exception as e:
                    if "429" in str(e): time.sleep(15); continue
                    break
        except: pass

    return ("**No Vision API configured.**\n\nAdd GROQ_API_KEY or GEMINI_API_KEY to your .env file "
            "to enable AI visual reports.\n\n- Free Groq key: console.groq.com (14,400 req/day)\n"
            "- Free Gemini key: aistudio.google.com"), "No API"

# ══ AI CHAT ═══════════════════════════════════════════════════════════════════
def chat_reply(user_msg, history, report, location):
    lang = st.session_state.get("lc","en")
    loc_note = f"User location: {location}." if location else ""
    lang_note = f"Respond in {lang} language." if lang != "en" else ""
    loc_enc = location.replace(" ","+") if location else ""
    maps_link = f"\n\n🏥 **Find doctors near {location}:** [Search Google Maps](https://www.google.com/maps/search/hospital+near+{loc_enc})" if location else ""

    system = f"""You are MediVision AI — a helpful medical imaging and health assistant. {lang_note} {loc_note}

RULES:
1. Always give a specific, helpful answer. Never refuse with "I can't provide medical advice" without first giving useful information.
2. For symptoms/conditions/treatments: give thorough educational info — causes, common treatments, medications typically used, lifestyle advice — then add disclaimer.
3. For doctor/hospital questions: provide Google Maps links for their location + specialist type to see.
4. Reference the image report when relevant.
5. End every medical answer with: "⚠️ Educational only. Please consult a qualified doctor. Not a prescription."
6. Use markdown for clarity. Be warm and genuinely helpful.

Image report context: {report or 'No image analyzed yet.'}"""

    msgs = history + [{"role":"user","content":user_msg}]

    # Try Groq text
    key = os.getenv("GROQ_API_KEY")
    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            for model in ["llama-3.3-70b-versatile","llama3-70b-8192","mixtral-8x7b-32768"]:
                try:
                    r = client.chat.completions.create(
                        model=model, max_tokens=1200, temperature=0.5,
                        messages=[{"role":"system","content":system}]+msgs)
                    reply = r.choices[0].message.content.strip()
                    if location and any(w in user_msg.lower() for w in
                        ["treatment","doctor","hospital","symptom","medicine","pain","cure","clinic"]):
                        reply += maps_link
                    return reply
                except: continue
        except: pass

    # Try Gemini text
    key = os.getenv("GEMINI_API_KEY")
    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            mn = st.session_state.get("_gm","models/gemini-1.5-flash")
            history_txt = "\n".join(f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in msgs)
            for _ in range(2):
                try:
                    r = genai.GenerativeModel(mn).generate_content(f"{system}\n\n{history_txt}\nAI:")
                    reply = r.text.strip()
                    if location and any(w in user_msg.lower() for w in ["treatment","doctor","symptom"]):
                        reply += maps_link
                    return reply
                except Exception as e:
                    if "429" in str(e): time.sleep(12); continue
                    break
        except: pass

    # Offline fallback
    ql = user_msg.lower()
    if any(x in ql for x in ["pneumonia","chest infection"]):
        return f"**Pneumonia Treatment (Educational)**\n\n- Bacterial: Antibiotics (amoxicillin, azithromycin)\n- Viral: Rest, antivirals if prescribed\n- Supportive: Fluids, paracetamol for fever, rest\n- Severe: Hospital, IV antibiotics, oxygen\n\n⚠️ Educational only. Consult a doctor.{maps_link}"
    if any(x in ql for x in ["tb","tuberculosis"]):
        return f"**TB Treatment (Educational)**\n\nDOTS therapy: Isoniazid + Rifampicin + Pyrazinamide + Ethambutol for 6-9 months. Available FREE at govt hospitals.\n\n⚠️ Educational only. Consult a doctor.{maps_link}"
    if any(x in ql for x in ["fracture","broken"]):
        return f"**Fracture Treatment (Educational)**\n\n- Immobilization: Cast/splint 4-12 weeks\n- Pain: Paracetamol/ibuprofen\n- Surgery: Plates/screws for complex fractures\n- Physiotherapy after healing\n\n⚠️ Educational only. Consult an orthopedic surgeon.{maps_link}"
    if any(x in ql for x in ["doctor","hospital","clinic","nearby","near me"]):
        loc = location or "your area"
        le  = loc.replace(" ","+")
        return f"**Finding Doctors Near {loc}**\n\n- [Hospitals]( https://www.google.com/maps/search/hospital+near+{le})\n- [Radiologists](https://www.google.com/maps/search/radiologist+near+{le})\n- [Specialists](https://www.google.com/maps/search/specialist+doctor+near+{le})\n\n⚠️ Please consult a qualified doctor."
    return f"I can help with medical questions, image findings, treatments, and finding doctors near you.\n\nTry asking:\n- *What does this X-ray show?*\n- *Treatment for pneumonia?*\n- *Find a hospital near me*{maps_link}"

# ══ HISTORY UTILS ═════════════════════════════════════════════════════════════
HISTORY_FILE = "medivision_history.json"

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f: return json.load(f)
    except: pass
    return []

def save_to_history(fname, report, source):
    hist = load_history()
    hist.insert(0, {"id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "date": datetime.now().strftime("%d %b %Y, %H:%M"),
                    "filename": fname, "report": report, "source": source})
    hist = hist[:20]  # keep last 20
    try:
        with open(HISTORY_FILE,"w") as f: json.dump(hist,f)
    except: pass

def ai_status():
    if os.getenv("GROQ_API_KEY"): return True, "Groq"
    if os.getenv("GEMINI_API_KEY"): return True, "Gemini"
    return False, "No API"

# ══ SESSION STATE ══════════════════════════════════════════════════════════════
for k,v in {"chat":[],"report":None,"last_file":None,"cache":None,
            "lc":"en","ln":"English","loc":""}.items():
    if k not in st.session_state: st.session_state[k]=v

# ══ SIDEBAR ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='padding:.4rem 0 1rem'><div style='font-size:1.4rem;font-weight:700;color:#f0f9ff'>🩺 {T('title')}</div><div style='font-size:.78rem;color:#475569'>{T('sub')}</div></div>", unsafe_allow_html=True)

    ln = st.selectbox("🌐 Language", list(LANGS.keys()),
                      index=list(LANGS.keys()).index(st.session_state.ln),
                      key="lang_sel")
    if ln != st.session_state.ln:
        st.session_state.ln=ln; st.session_state.lc=LANGS[ln]
        st.session_state.cache=None; st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio("", [T("upload"),T("chat"),T("history"),T("how")], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    alpha = st.slider("Overlay Opacity", 0.1, 0.7, 0.35, 0.05)
    color = st.selectbox("Mask Color", ["Red","Cyan","Green","Yellow"])
    CMAP  = {"Red":(0,60,255),"Cyan":(255,200,0),"Green":(0,220,60),"Yellow":(0,230,255)}

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div class='sl'>📍 {T('loc')}</div>", unsafe_allow_html=True)
    new_loc = st.text_input("loc","",placeholder="Chennai, Mumbai...",label_visibility="collapsed")
    if new_loc != st.session_state.loc: st.session_state.loc = new_loc

    st.markdown("<hr>", unsafe_allow_html=True)
    ok, label = ai_status()
    st.markdown(f'<span class="badge {"bg" if ok else "bo"}">{"✓" if ok else "⚠"} {label}</span>', unsafe_allow_html=True)
    if not ok: st.caption("Add GROQ_API_KEY to .env (free: console.groq.com)")

    st.markdown(f"<hr><div class='disc'>{T('disc')}</div>", unsafe_allow_html=True)

def hdr(title, sub=""):
    st.markdown(f"<h1 class='ht'>{title}</h1>{'<p class=hs>'+sub+'</p>' if sub else ''}<hr>", unsafe_allow_html=True)

# ══ PAGE 1 — UPLOAD ════════════════════════════════════════════════════════════
if T("upload") in page:
    hdr(T("upload"), "Upload any medical image — AI will describe what it sees.")

    c1, c2 = st.columns([3,2], gap="large")
    with c1:
        uploaded = st.file_uploader("img", type=["png","jpg","jpeg","bmp","tif","tiff"], label_visibility="collapsed")
    with c2:
        st.markdown("<div class='card ca' style='margin-top:.5rem'><div class='sl'>Supported Formats</div><div style='color:#94a3b8;font-size:.86rem;line-height:1.9'>🖼️ PNG / JPG / BMP / TIFF<br>🏥 X-Ray, MRI, CT, Ultrasound<br>🤖 AI reads the actual image visually</div></div>", unsafe_allow_html=True)

    if uploaded:
        if uploaded.name != st.session_state.last_file:
            st.session_state.cache=None; st.session_state.last_file=uploaded.name
            st.session_state.report=None; st.session_state.chat=[]

        img = load_img(uploaded)
        if img is None: st.error("Could not read image. Upload PNG or JPG."); st.stop()

        with st.spinner(T("analyzing")):
            mask = segment(img)
        ov_img = make_overlay(img, mask, CMAP[color], alpha)

        c1,c2,c3 = st.columns(3,gap="medium")
        for col, label, image in [(c1,"Original",cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),
                                   (c2,"Mask",mask),(c3,"Overlay",cv2.cvtColor(ov_img,cv2.COLOR_BGR2RGB))]:
            with col:
                st.markdown(f"<div class='sl'>{label}</div>", unsafe_allow_html=True)
                st.image(image, use_column_width=True, clamp=True)

        st.markdown(f"<hr><div class='sl'>{T('report')}</div>", unsafe_allow_html=True)

        if not st.session_state.cache:
            with st.spinner(T("analyzing")):
                rep, src = vision_report(img, uploaded.name)
                st.session_state.cache=(rep,src); st.session_state.report=rep
                save_to_history(uploaded.name, rep, src)
                if not st.session_state.chat:
                    st.session_state.chat=[{"role":"assistant","content":
                        f"I've analyzed **{uploaded.name}**. The visual report is ready. Ask me anything about the findings, symptoms, treatments, or I can help find nearby doctors!"}]

        rep, src = st.session_state.cache
        bc = "bg" if "Vision" in src else "bo"
        st.markdown(f"<div class='card ca'><div style='color:#cbd5e1;font-size:.91rem;line-height:1.85'>{rep.replace(chr(10),'<br>')}</div><div style='font-size:.68rem;color:#475569;margin-top:.7rem;border-top:1px solid #1e2d45;padding-top:.5rem'><span class='badge {bc}'>{src}</span> · {T('disc')}</div></div>", unsafe_allow_html=True)
        st.success(T("done"))

        if st.session_state.loc:
            le = st.session_state.loc.replace(" ","+")
            st.markdown(f"<div class='card dlink'><b>🏥 {T('find')} — {st.session_state.loc}</b><br>"
                        f"<a href='https://www.google.com/maps/search/hospital+near+{le}' target='_blank'>Hospitals</a> · "
                        f"<a href='https://www.google.com/maps/search/radiologist+near+{le}' target='_blank'>Radiologists</a> · "
                        f"<a href='https://www.google.com/maps/search/specialist+doctor+near+{le}' target='_blank'>Specialists</a></div>",
                        unsafe_allow_html=True)

# ══ PAGE 2 — CHATBOT ═══════════════════════════════════════════════════════════
elif T("chat") in page:
    hdr(T("chat"), "Ask about your image, symptoms, treatments, or find nearby doctors.")

    if not st.session_state.chat:
        st.markdown("<div class='card' style='text-align:center;padding:2.5rem'><div style='font-size:2.2rem'>🩺</div><div style='color:#f0f9ff;font-weight:600;font-size:1rem;margin-top:.5rem'>MediVision AI</div><div style='color:#64748b;font-size:.86rem;margin-top:.4rem'>Upload an image first, or ask anything about medical imaging.</div></div>", unsafe_allow_html=True)
    else:
        for m in st.session_state.chat:
            if m["role"]=="user":
                st.markdown(f"<div class='lu'>You</div><div class='cu'>{m['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='la'>🩺 MediVision AI</div><div class='ca2'>{m['content'].replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)

    st.markdown("<hr><div class='sl'>Quick Questions</div>", unsafe_allow_html=True)
    quick = ["What does this image show?","What are possible conditions?","What treatments are available?",
             "What symptoms should I watch for?","Find a doctor near me","Difference between MRI and CT?"]
    for i,qp in enumerate(quick):
        with st.columns(3)[i%3]:
            if st.button(qp, key=f"q{i}"):
                st.session_state.chat.append({"role":"user","content":qp})
                with st.spinner(T("thinking")):
                    r = chat_reply(qp, st.session_state.chat[:-1],
                                   st.session_state.report or "", st.session_state.loc)
                st.session_state.chat.append({"role":"assistant","content":r})
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    with st.form("cf", clear_on_submit=True):
        ci,cb = st.columns([5,1])
        with ci: usr = st.text_input("m", placeholder=T("ph"), label_visibility="collapsed")
        with cb: go  = st.form_submit_button(T("send"))
    if go and usr.strip():
        st.session_state.chat.append({"role":"user","content":usr.strip()})
        with st.spinner(T("thinking")):
            r = chat_reply(usr.strip(), st.session_state.chat[:-1],
                           st.session_state.report or "", st.session_state.loc)
        st.session_state.chat.append({"role":"assistant","content":r})
        st.rerun()
    if st.session_state.chat:
        if st.button(T("clear")): st.session_state.chat=[]; st.rerun()

# ══ PAGE 3 — HISTORY ═══════════════════════════════════════════════════════════
elif T("history") in page:
    hdr("📋 Analysis History", "Your past image reports — tap any to review.")

    hist = load_history()
    if not hist:
        st.markdown("<div class='card' style='text-align:center;padding:2rem;color:#64748b'>No history yet. Analyze an image to start building your history.</div>", unsafe_allow_html=True)
    else:
        search = st.text_input("🔍 Search history", placeholder="Search by filename or finding...", key="hsearch")
        filtered = [h for h in hist if not search or search.lower() in h["filename"].lower() or search.lower() in h["report"].lower()]

        st.markdown(f"<div style='color:#475569;font-size:.8rem;margin-bottom:.6rem'>{len(filtered)} record(s) found</div>", unsafe_allow_html=True)

        for h in filtered:
            bc = "bg" if "Vision" in h.get("source","") else "bo"
            with st.expander(f"🗂 {h['filename']}  ·  {h['date']}"):
                st.markdown(f"<span class='badge {bc}'>{h.get('source','Unknown')}</span>", unsafe_allow_html=True)
                st.markdown(h["report"])
                c1,c2 = st.columns(2)
                with c1:
                    if st.button("Load into chatbot", key=f"load_{h['id']}"):
                        st.session_state.report = h["report"]
                        st.session_state.chat=[{"role":"assistant","content":
                            f"Loaded past report for **{h['filename']}** from {h['date']}. Ask me anything about it!"}]
                        st.success("Loaded! Go to 💬 AI Chatbot.")
                with c2:
                    if st.button("Delete", key=f"del_{h['id']}"):
                        all_h = load_history()
                        all_h = [x for x in all_h if x["id"] != h["id"]]
                        try:
                            with open(HISTORY_FILE,"w") as f: json.dump(all_h,f)
                        except: pass
                        st.rerun()

        if st.button("🗑️ Clear All History"):
            try: os.remove(HISTORY_FILE)
            except: pass
            st.rerun()

# ══ PAGE 4 — HOW IT WORKS ══════════════════════════════════════════════════════
elif T("how") in page:
    hdr(T("how"), "Vision AI pipeline and setup guide.")

    for n,t,d in [
        ("1","Vision AI Report","The actual image is sent to Groq Vision (Llama 4) or Gemini Vision. The AI describes what it visually sees — no pixel statistics used."),
        ("2","Segmentation Overlay","CLAHE + Otsu threshold generates a visual mask overlay. Used for display only, not the report."),
        ("3","Medical Chatbot","Gives real answers about symptoms, treatments, medications. Never refuses — always provides helpful educational information + disclaimer."),
        ("4","Doctor Finder","Enter your city in the sidebar. Google Maps links for hospitals, radiologists, and specialists appear automatically."),
        ("5","Analysis History","Every report is saved locally. Search, review, or load any past analysis into the chatbot."),
        ("6","Multilanguage","12 languages supported. AI reports and chatbot respond in the selected language."),
    ]:
        st.markdown(f"<div class='card ca' style='display:flex;gap:1rem;align-items:flex-start'><div style='min-width:2rem;height:2rem;background:#0ea5e9;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#0b0f1a;flex-shrink:0'>{n}</div><div><div style='font-weight:600;color:#f0f9ff;margin-bottom:.2rem'>{t}</div><div style='color:#94a3b8;font-size:.87rem;line-height:1.6'>{d}</div></div></div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.code("""# Install
pip install streamlit opencv-python-headless numpy python-dotenv groq google-generativeai

# .env file
GROQ_API_KEY=your_key     # free at console.groq.com — 14,400 req/day, Llama 4 Vision
GEMINI_API_KEY=your_key   # free at aistudio.google.com — gemini-1.5-flash Vision

# Run
streamlit run app.py""", language="bash")