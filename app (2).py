import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Smart Beehive", page_icon="🐝", layout="centered")

GREEN, HONEY, CREAM = "#2E5E3E", "#E8A317", "#FBF7EA"
st.markdown(
    f"""<style>
.stApp {{background:{CREAM};}}
h1,h2,h3 {{color:{GREEN};}}
.card {{background:#fff;border-radius:14px;padding:12px 14px;margin:6px 0;
        border:1px solid #eadfb8;box-shadow:0 1px 3px rgba(0,0,0,.06);}}
.big {{font-size:1.6rem;font-weight:700;color:{GREEN};}}
.lbl {{font-size:.8rem;color:#7a6a3a;}}
.stButton>button, .stFormSubmitButton>button {{background:{GREEN};color:#fff;border-radius:10px;border:0;}}
.stTabs [aria-selected="true"] {{color:{GREEN};}}
</style>""",
    unsafe_allow_html=True,
)


def sb():
    if "sb" not in st.session_state:
        st.session_state.sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])
    return st.session_state.sb


def card(label, value):
    st.markdown(f'<div class="card"><div class="lbl">{label}</div><div class="big">{value}</div></div>',
                unsafe_allow_html=True)


def rows(table, hive_id, limit=500, order="ts"):
    r = sb().table(table).select("*").eq("hive_id", hive_id).order(order, desc=True).limit(limit).execute()
    return pd.DataFrame(r.data)


def auth_screen():
    st.title("🐝 Smart Beehive")
    mode = st.radio("Mode", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
    email = st.text_input("Email")
    pw = st.text_input("Password (min 6 characters)", type="password")
    name = st.text_input("Your name") if mode == "Register" else ""
    if st.button(mode):
        try:
            if mode == "Register":
                r = sb().auth.sign_up({"email": email, "password": pw, "options": {"data": {"full_name": name}}})
                if r.session is None:
                    st.success("Registered. Check your email to confirm, then log in.")
                    return
            else:
                r = sb().auth.sign_in_with_password({"email": email, "password": pw})
            st.session_state.user = r.user
            st.rerun()
        except Exception as e:
            st.error(str(e))


def extract_features(y, sr):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mean_spec = S.mean(axis=1)
    band = (freqs >= 50) & (freqs <= 2000)
    f = {
        "dominant_freq": float(freqs[band][np.argmax(mean_spec[band])]),
        "spectral_centroid": float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
        "spectral_bandwidth": float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()),
        "spectral_rolloff": float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean()),
        "spectral_flatness": float(librosa.feature.spectral_flatness(S=S).mean()),
        "rms_energy": float(librosa.feature.rms(S=S).mean()),
        "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y, hop_length=512).mean()),
        "spectral_energy": float((S ** 2).sum(axis=0).mean()),
    }
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    for k in range(13):
        f["mfcc%d_mean" % (k + 1)] = float(m[k].mean())
        f["mfcc%d_std" % (k + 1)] = float(m[k].std())
    return f, mean_spec, freqs


def tab_home(hive):
    r = rows("sensor_readings", hive["id"], limit=1)
    st.subheader(hive["name"])
    st.caption("Mode: " + hive["mode"])
    if r.empty:
        st.info("No readings yet. Run simulator.py to send data.")
    else:
        x = r.iloc[0]
        a, b = st.columns(2)
        with a:
            card("Temperature", f"{x['temperature']:.1f} °C")
            card("Weight", f"{x['weight_kg']:.2f} kg")
        with b:
            card("Humidity", f"{x['humidity']:.0f} %")
            card("Battery", "n/a" if pd.isna(x["battery_pct"]) else f"{x['battery_pct']:.0f} %")
    c1, c2, c3 = st.columns(3)
    for col, m in zip((c1, c2, c3), ("normal", "harvest", "maintenance")):
        if col.button(m.capitalize(), key="mode_" + m):
            sb().table("beehives").update({"mode": m}).eq("id", hive["id"]).execute()
            st.rerun()
    al = rows("alerts", hive["id"], limit=5)
    st.markdown("**Recent alerts**")
    if al.empty:
        st.write("No alerts.")
    else:
        for _, a in al.iterrows():
            st.warning(f"{a['message']}")


def tab_analytics(hive):
    df = rows("sensor_readings", hive["id"], limit=1000)
    if df.empty:
        st.info("No data yet.")
        return
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")
    st.markdown("**Temperature (°C)**")
    st.line_chart(df["temperature"], color=HONEY)
    st.markdown("**Humidity (%)**")
    st.line_chart(df["humidity"], color=GREEN)
    st.markdown("**Weight (kg)**")
    st.line_chart(df["weight_kg"], color=GREEN)
    au = rows("audio_analysis", hive["id"], limit=200)
    if not au.empty:
        au["ts"] = pd.to_datetime(au["ts"])
        st.markdown("**Bee activity probability (experimental model)**")
        st.line_chart(au.sort_values("ts").set_index("ts")["bee_activity_prob"], color=HONEY)


def tab_sound(hive):
    st.write("Upload a hive recording (.wav). The model only tells bee vs no-bee activity. "
             "It does not detect queen piping or swarming.")
    up = st.file_uploader("Hive audio", type=["wav"])
    if up is None:
        return
    try:
        import librosa
        y, sr = librosa.load(io.BytesIO(up.read()), sr=22050, mono=True, duration=30)
    except Exception as e:
        st.error("Could not read audio: " + str(e))
        return
    feats, spec, freqs = extract_features(y[: 5 * sr], sr)
    a, b = st.columns(2)
    with a:
        card("Dominant frequency", f"{feats['dominant_freq']:.0f} Hz")
    with b:
        card("Spectral centroid", f"{feats['spectral_centroid']:.0f} Hz")
    st.markdown("**FFT spectrum (first 5 s)**")
    keep = freqs <= 2000
    st.line_chart(pd.DataFrame({"magnitude": spec[keep]}, index=freqs[keep]), color=GREEN)
    mp, cp = "models/zenodo_bee_activity_model.pkl", "models/zenodo_feature_columns.json"
    if os.path.exists(mp) and os.path.exists(cp):
        import joblib
        model = joblib.load(mp)
        cols = json.load(open(cp))
        probs = []
        for i in range(0, len(y) - 5 * sr + 1, 5 * sr):
            f, _, _ = extract_features(y[i:i + 5 * sr], sr)
            probs.append(model.predict_proba(pd.DataFrame([f])[cols])[0][1])
        if probs:
            p = float(np.mean(probs))
            label = "bee activity" if p >= 0.5 else "no-bee / low activity"
            st.success(f"Experimental result: {label} (probability {p:.2f})")
            if st.button("Save this result"):
                sb().table("audio_analysis").insert({
                    "hive_id": hive["id"], "dominant_freq": feats["dominant_freq"],
                    "spectral_centroid": feats["spectral_centroid"], "rms_energy": feats["rms_energy"],
                    "spectral_flatness": feats["spectral_flatness"], "bee_activity_prob": p,
                    "label": label, "model_version": "zenodo_rf_v1"}).execute()
                st.success("Saved.")
    else:
        st.info("Model files not found. Put zenodo_bee_activity_model.pkl and "
                "zenodo_feature_columns.json in a models/ folder next to app.py.")


def tab_alerts(hive):
    al = rows("alerts", hive["id"], limit=100)
    if al.empty:
        st.write("No alerts.")
        return
    for _, a in al.iterrows():
        st.markdown(f"**{a['type']}**  \n{a['message']}  \n`{str(a['ts'])[:16]}`")
        if not a["acknowledged"]:
            if st.button("Acknowledge", key="ack%s" % a["id"]):
                sb().table("alerts").update({"acknowledged": True}).eq("id", int(a["id"])).execute()
                st.rerun()
        st.divider()


def tab_harvest(hive):
    r = rows("sensor_readings", hive["id"], limit=1)
    cur = float(r.iloc[0]["weight_kg"]) if not r.empty else 0.0
    st.write("Tip: switch the hive to Harvest mode (Home tab) before removing honey.")
    with st.form("harvest"):
        before = st.number_input("Weight before (kg)", value=cur, step=0.1)
        after = st.number_input("Weight after (kg)", value=max(cur - 1.0, 0.0), step=0.1)
        yld = st.number_input("Actual honey yield (kg)", value=0.0, step=0.1)
        notes = st.text_area("Notes")
        if st.form_submit_button("Save harvest"):
            sb().table("harvest_events").insert({
                "hive_id": hive["id"], "weight_before": before, "weight_after": after,
                "honey_yield_kg": yld, "notes": notes}).execute()
            sb().table("beehives").update({"mode": "normal"}).eq("id", hive["id"]).execute()
            st.success("Harvest saved and mode set back to normal.")
    h = rows("harvest_events", hive["id"], limit=50, order="harvest_date")
    if not h.empty:
        st.dataframe(h[["harvest_date", "weight_before", "weight_after", "weight_removed",
                        "honey_yield_kg", "notes"]], hide_index=True)


def tab_hives(hives):
    for h in hives:
        st.markdown(f'<div class="card"><b>{h["name"]}</b><br>Code: {h["hive_code"]}<br>'
                    f'{h["location"] or ""}</div>', unsafe_allow_html=True)
    with st.form("add"):
        st.markdown("**Add a beehive**")
        code = st.text_input("Hive code (example: HIVE_A1)")
        name = st.text_input("Hive name")
        loc = st.text_input("Location")
        if st.form_submit_button("Register hive"):
            try:
                k = sb().rpc("register_hive", {"p_code": code, "p_name": name, "p_location": loc}).execute().data
                st.success("Hive registered. Copy this device key now, it is shown only once:")
                st.code(k)
            except Exception as e:
                st.error(str(e))


def tab_settings(hive):
    t = hive["thresholds"]
    with st.form("thr"):
        tmin = st.number_input("Min temperature (°C)", value=float(t.get("temp_min", 15)))
        tmax = st.number_input("Max temperature (°C)", value=float(t.get("temp_max", 40)))
        hmax = st.number_input("Max humidity (%)", value=float(t.get("humidity_max", 85)))
        bmin = st.number_input("Low battery (%)", value=float(t.get("battery_min", 20)))
        wd = st.number_input("Weight drop alert (kg)", value=float(t.get("weight_drop_kg", 1.0)))
        if st.form_submit_button("Save thresholds"):
            sb().table("beehives").update({"thresholds": {
                "temp_min": tmin, "temp_max": tmax, "humidity_max": hmax,
                "battery_min": bmin, "weight_drop_kg": wd}}).eq("id", hive["id"]).execute()
            st.success("Saved.")
    st.write("Logged in as:", st.session_state.user.email)
    if st.button("Log out"):
        sb().auth.sign_out()
        st.session_state.clear()
        st.rerun()


def main():
    hives = sb().table("beehives").select("*").order("created_at").execute().data
    hive = None
    if hives:
        idx = st.selectbox("Hive", range(len(hives)), format_func=lambda i: hives[i]["name"])
        hive = hives[idx]
    names = ["Home", "My Hives", "Analytics", "Sound", "Alerts", "Harvest", "Settings"]
    tabs = st.tabs(names)
    with tabs[1]:
        tab_hives(hives)
    funcs = {0: tab_home, 2: tab_analytics, 3: tab_sound, 4: tab_alerts, 5: tab_harvest, 6: tab_settings}
    for i, fn in funcs.items():
        with tabs[i]:
            if hive is None:
                st.info("Add a hive in the My Hives tab first.")
            else:
                fn(hive)


if "user" not in st.session_state:
    auth_screen()
else:
    main()
