import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import time

st.set_page_config(page_title="Chord Detector", layout="wide")

science_info = {
    'C': {'desc': 'التردد الأساسي لهذه النغمة حوالي 261.63 هرتز، وهي تعتبر نقطة الاستقرار الأساسية في السلم الموسيقي.', 'feel': 'تعطي إحساساً بالهدوء والاكتمال.'},
    
    'C#': {'desc': 'تردد هذه النغمة يقارب 277.18 هرتز وتستخدم غالباً كنغمة انتقالية بين المقامات.', 'feel': 'تضيف توتراً موسيقياً خفيفاً.'},
    
    'D': {'desc': 'ترددها يقارب 293.66 هرتز وتتميز بوضوحها في الطيف الصوتي.', 'feel': 'تعطي شعوراً بالحركة والطاقة.'},
    
    'D#': {'desc': 'ترددها حوالي 311.13 هرتز وتحتوي على توافقيات واضحة في التحليل الطيفي.', 'feel': 'تميل إلى إعطاء عمق موسيقي.'},
    
    'E': {'desc': 'ترددها يقارب 329.63 هرتز وتعتبر من النغمات المستقرة في الهارموني.', 'feel': 'تعطي إحساساً بالسكينة والوضوح.'},
    
    'F': {'desc': 'ترددها حوالي 349.23 هرتز ويظهر رنينها بوضوح في مخطط الطيف.', 'feel': 'تعطي إحساساً بالقوة والاتساع.'},
    
    'F#': {'desc': 'ترددها يقارب 369.99 هرتز وتتميز بتركيب توافقي غني.', 'feel': 'تعطي طابعاً غامضاً قليلاً.'},
    
    'G': {'desc': 'ترددها حوالي 392 هرتز وتظهر بطاقة عالية في التحليل الترددي.', 'feel': 'مرتبطة بالنشاط والإشراق.'},
    
    'G#': {'desc': 'ترددها يقارب 415.30 هرتز وغالباً ما تستخدم في الانتقالات اللحنية.', 'feel': 'تعطي إحساساً عاطفياً أو حنينياً.'},
    
    'A': {'desc': 'ترددها القياسي 440 هرتز وتستخدم كمرجع عالمي لضبط الآلات الموسيقية.', 'feel': 'نغمة متوازنة وواضحة.'},
    
    'A#': {'desc': 'ترددها يقارب 466.16 هرتز وتتميز بطاقة صوتية مرتفعة نسبياً.', 'feel': 'تضيف قوة وإثارة للحن.'},
    
    'B': {'desc': 'ترددها حوالي 493.88 هرتز وتتميز بتوتر موسيقي يدفع للانتقال إلى النغمة التالية.', 'feel': 'تعطي إحساساً بالتوقع والانتظار.'}
}
@st.cache_resource
def load_assets():
    try:
        data = joblib.load("chord_pipeline.pkl")
        return data["pipeline"], data["label_encoder"]
    except:
        return None, None

def draw_piano(root_note):
    NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    fig, ax = plt.subplots(figsize=(8, 3))
    for i, note in enumerate(NOTES):
        color = "#FFD700" if note == root_note else "white"
        rect = plt.Rectangle((i, 0), 1, 1, color=color, ec="black", lw=2)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.4, note, ha="center", va="center", fontweight='bold', fontsize=12)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig


head_col1, head_col2 = st.columns([1, 9])
with head_col1:
    st.image("https://img.icons8.com/fluency/96/000000/audio-wave.png", width=60)
with head_col2:
    st.title("Musical Chord Detector")

st.markdown("---")

pipeline, le = load_assets()

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.results = {}

#  رفع الملفات
col_up1, col_up2, col_up3 = st.columns([1,2,1])
with col_up2:
    upload_file = st.file_uploader("قم برفع ملف الصوت (WAV/MP3)", type=["wav", "mp3"])
    if upload_file:
        st.audio(upload_file)
        if st.button(" ابدأ التحليل "):
            if not pipeline:
                st.error(" ملف الموديل غير موجود!")
            else:
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(upload_file.read())
                    path = tmp.name
                
                y, sr = librosa.load(path,sr=22050, duration=4.0)
                chroma = librosa.feature.chroma_cens(y=y, sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr ,n_fft=512)
                combined = np.vstack((chroma, contrast, mfcc))
                feats = np.concatenate([np.mean(combined.T, 0),
                                        np.std(combined.T, 0),
                                        ]).reshape(1, -1)
                
                pred_num = pipeline.predict(feats)[0]
                chord_type = le.inverse_transform([pred_num])[0]
                NOTES_LIST = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                chroma_mean = np.mean(chroma, axis=1)
                root_note_index = np.argmax(chroma_mean)
                root_note = NOTES_LIST[root_note_index]
                root_note = NOTES_LIST[np.argmax(np.mean(chroma, axis=1))]
                probs = pipeline.predict_proba(feats)[0] 
                confidence = np.max(probs)
                chords = le.classes_  
                st.session_state.results = {
                    'chord': f"{root_note} {chord_type}",
                    'root': root_note,
                    'type': chord_type,
                    'y': y,
                    'sr': sr,
                    'chroma': chroma,
                    'probs': probs,
                    'chords': chords,
                    
            }
                st.session_state.analyzed = True

#  عرض النتائج 
if st.session_state.analyzed:
    res = st.session_state.results
    st.divider()
    
    st.markdown(f"###  النتيجة المكتشفة: `{res['chord']}`")
    m1, m2, m3 = st.columns(3)
    m1.metric("النغمة الأساسية", res['root'])
    if 'confidence' in res:
        m2.metric("دقة التوقع", f"{res['confidence']:.2%}")
    
    st.write("") 

    tab1, tab2, tab3 = st.tabs([" التحليل البصري", " لوحة المفاتيح" ," التفسير العلمي"])

    with tab1:
        st.markdown("#### التحليل الموجي والطيفي")
        c1, c2 = st.columns(2)
        
        #  Waveform
        with c1:
            st.caption("Waveform (الموجة الصوتية)")
            fig_w, ax_w = plt.subplots(figsize=(5, 3))
            librosa.display.waveshow(res['y'], sr=res['sr'], ax=ax_w, color='#00d1b2', alpha=0.8)
            ax_w.set_axis_off() 
            st.pyplot(fig_w, use_container_width=True)
            
        #  Spectrogram
        with c2:
            st.caption("Spectrogram (مخطط الطيف)")
            fig_s, ax_s = plt.subplots(figsize=(5, 3))
            S_db = librosa.amplitude_to_db(np.abs(librosa.stft(res['y'])), ref=np.max)
            img = librosa.display.specshow(S_db, sr=res['sr'], x_axis='time', y_axis='log', ax=ax_s, cmap='magma')
            ax_s.set_axis_off()
            st.pyplot(fig_s, use_container_width=True)
            st.markdown("#### احتمالات الكوردات")
        
        _, center_plot, _ = st.columns([1, 5, 1]) 
        
        with center_plot:
            fig_p, ax_p = plt.subplots(figsize=(8, 3))
            colors = ['#FFD700' if c == res['type'] else '#00d1b2' for c in res['chords']]
            ax_p.bar(range(len(res['chords'])), res['probs'], color=colors, alpha=0.8)
            ax_p.set_ylabel("Probability")
            ax_p.set_ylim(0, 1)
            ax_p.set_xticks(range(len(res['chords'])))
            ax_p.set_xticklabels(res['chords'], rotation=45, ha="right")
            st.pyplot(fig_p, use_container_width=True)
    
    with tab2:
        st.markdown("### موقع النغمة على البيانو")
        _, center_col, _ = st.columns([1, 3, 1])
        with center_col:
            st.pyplot(draw_piano(res['root']))

    with tab3:
        info = science_info.get(res['root'], {'desc': 'نغمة موسيقية.', 'feel': 'جزء من السلم الموسيقي.'})
        
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("####  الخصائص الفيزيائية")
            st.write(info['desc'])
            
        with c2:
            st.markdown("####  التأثير الشعوري")
            st.markdown(f"""
            > **{info['feel']}**
            """, help="هذا التحليل مبني على السيكولوجيا الموسيقية للنغمات.")
    
        st.divider()
            