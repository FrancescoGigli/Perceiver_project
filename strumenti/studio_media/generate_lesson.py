"""Build the source-faithful QKV lesson with local neural speech and SVG animation.

Run with the isolated Python runtime used for neural_tts.py. Requires ffmpeg on PATH.
No model, dependency, cache, temporary WAV or rendered frame enters the repository.
"""
from __future__ import annotations

import argparse
from html import escape, unescape
import io
import json
import math
from pathlib import Path
import re
import subprocess
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import resvg_py
import soundfile as sf

from neural_tts import synthesize

REPO = Path(__file__).resolve().parents[2]
MEDIA = REPO / "sito" / "studio" / "media"
INDIGO = "#1a237e"
ORANGE = "#e65100"
MUTED = "#6c7081"
FONT = "Segoe UI"


def svg_text(x, y, text, size=26, fill=INDIGO, weight=400, anchor="start", family=FONT):
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(text)}</text>'


def box(x, y, width, height, fill="#ffffff", stroke="#dce0ee", radius=12, sw=2):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def scene_svg(step, index, total):
    stage = step["stage"]
    labels = {"query": "Q · la domanda", "key": "K · l'etichetta", "value": "V · il contenuto", "scores": "La domanda incontra le etichette", "weights": "Una media pesata dei contenuti", "read": "K e V vengono dall'input", "output": "Da M² a MN"}
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img" aria-labelledby="title desc">',f'<title id="title">{escape(labels[stage])}</title>', '<desc id="desc">Schema didattico di cross-attention, con due latenti e quattro elementi di input. Il numero di elementi e gli accenti grafici non rappresentano i pesi del modello.</desc>', '<rect width="1280" height="720" fill="#f7f8fc"/>', '<rect width="1280" height="10" fill="#1a237e"/>', svg_text(54, 45, "PERCEIVER · LEZIONE 1.2", 18, MUTED, 600), svg_text(54, 95, labels[stage], 36, INDIGO, 700), svg_text(1225, 45, f"{index + 1} / {total}", 20, ORANGE, 700, "end"), box(42, 123, 1196, 304)]
    s += [svg_text(130, 160, "Latenti L", 24, INDIGO, 700),svg_text(1030, 160, "Input X", 24, INDIGO, 700, "middle")]
    for n in range(2):
        y = 189 + 98 * n
        active = stage == "query"
        s += [box(100, y, 155, 67, "#e8eaf6", INDIGO),svg_text(177, y + 41, f"Latente {n + 1}", 24, INDIGO,600,"middle"),box(295, y, 105, 67, "#fff3e0" if active else "#f3f4f8", ORANGE if active else "#dce0ee",sw=3 if active else 2),svg_text(347,y+43,"Q",32,ORANGE if active else INDIGO,700,"middle"),f'<path d="M255 {y+33} H295" stroke="{ORANGE if active else INDIGO}" stroke-width="3" fill="none"/>']
    for n in range(4):
        y = 182 + 51 * n
        s += [box(919,y,226,42,"#f3f4f8", "#dce0ee",8),svg_text(931,y+28,f"{n+1}",17,MUTED),box(962,y,73,42,"#fff3e0" if stage in {"key","scores","read"} else "#e8eaf6",ORANGE if stage in {"key","scores","read"} else INDIGO,8),svg_text(999,y+29,"K",25,ORANGE if stage in {"key","scores","read"} else INDIGO,700,"middle"),box(1048,y,73,42,"#fff3e0" if stage in {"value","weights","read"} else "#e8eaf6",ORANGE if stage in {"value","weights","read"} else INDIGO,8),svg_text(1084,y+29,"V",25,ORANGE if stage in {"value","weights","read"} else INDIGO,700,"middle")]
    if stage in {"scores", "weights", "output"}:
        for n in range(4):
            y=203+51*n
            s += [f'<path d="M400 222 C610 222 735 {y} 919 {y}" stroke="#d9ddee" stroke-width="2" fill="none"/>']
        s += [svg_text(630,190,"A = softmax(dots)" if stage == "weights" else "QKᵀ",25,INDIGO,700,"middle")]
        for row in range(2):
            for col in range(4):
                s += [box(545+40*col,211+42*row,33,33,"#e8eaf6",INDIGO,5)]
        s += [svg_text(623,315,"N×M",24,ORANGE,700,"middle")]
    else:
        s += [svg_text(640,245,"domanda",25,INDIGO,600,"middle"), svg_text(640,285,"etichetta → contenuto",23,MUTED,400,"middle")]
    if stage == "weights":
        for n in range(4):
            y = 203 + 51*n
            s += [f'<path d="M1048 {y} C825 {y} 825 358 735 358" stroke="#efc6aa" stroke-width="2" fill="none"/>']
        s += [f'<path d="M510 358 C430 358 370 222 255 222" stroke="{ORANGE}" stroke-width="3" fill="none"/>',f'<path d="M255 222 L270 214 L270 230 Z" fill="{ORANGE}"/>',box(510,335,225,48,"#fff3e0",ORANGE,10),svg_text(622,366,"media pesata",24,ORANGE,700,"middle")]
    elif stage == "output":
        s += [box(500,329,250,54,"#fff3e0",ORANGE),svg_text(625,365,"O(M²) → O(MN)",26,ORANGE,700,"middle")]
    s += [svg_text(66,408,"Schema didattico · nessun peso numerico del modello",17,MUTED),box(42,443,1196,125,"#e8eaf6","#d7dcee",10),svg_text(66,468,"IL TUO CODICE · progetto/src/perceiver/attention.py",17,INDIGO,600)]
    code_y = 502
    for line in step["code"]:
        wrapped = textwrap.wrap(line["text"], width=87, break_long_words=False, break_on_hyphens=False)
        for j, part in enumerate(wrapped):
            prefix = str(line["line"]) if j == 0 else "  "
            s += [svg_text(67,code_y,prefix,20,ORANGE,600,family="Consolas"),svg_text(113,code_y,part,20,INDIGO,family="Consolas")]
            code_y += 26
    words = textwrap.wrap(step["text"],width=79,break_long_words=False,break_on_hyphens=False)
    caption_y = 617 if len(words)<3 else 602
    for word_line in words:
        s += [svg_text(640,caption_y,word_line,28,INDIGO,600,"middle")]
        caption_y += 37
    s += [svg_text(54,694,"Testo originale · Cap. 1.2 della tua lezione",16,MUTED),'<rect x="0" y="710" width="1280" height="10" fill="#dce0ee"/>','</svg>']
    return "".join(s)


def stamp(seconds, sep="."):
    ms=round(seconds*1000)
    h, ms=divmod(ms,3600000)
    m, ms=divmod(ms,60000)
    sec, ms=divmod(ms,1000)
    return f"{h:02}:{m:02}:{sec:02}{sep}{ms:03}"


def verify_sources(data):
    lesson = (REPO / "sito/lezione/index.html").read_text(encoding="utf-8")
    plain = unescape(re.sub(r"<[^>]+>", "", lesson))
    plain = re.sub(r"\s+", " ", plain)
    code_lines = (REPO / "progetto/src/perceiver/attention.py").read_text(encoding="utf-8").splitlines()
    for step in data["steps"]:
        expected=re.sub(r"\s+", " ",step["text"])
        assert expected in plain, f"Source text changed: {expected}"
        for line in step["code"]:
            assert code_lines[line["line"]-1].strip()==line["text"],line


def build(runtime, speed=.86, fps=20, reuse_audio=False):
    MEDIA.mkdir(parents=True, exist_ok=True)
    temporary = runtime / "lesson-build"
    temporary.mkdir(parents=True,exist_ok=True)
    data_path=MEDIA / "lesson-qkv.json"
    data=json.loads(data_path.read_text(encoding="utf-8-sig"))
    verify_sources(data)
    rate=24000
    segments=[]
    position=.45
    segments.append(np.zeros(round(rate*position),dtype=np.float32))
    for i,step in enumerate(data["steps"]):
        wav=temporary/f"segment-{i:02}.wav"
        if not reuse_audio or not wav.exists():
            result=synthesize(step["text"],wav,"im_nicola",speed,runtime)
            print(json.dumps({"segment":i,**result}),flush=True)
        samples,rate=sf.read(str(wav),dtype="float32")
        pause=.50
        step["start"]=round(position,3)
        step["end"]=round(position+len(samples)/rate+pause,3)
        segments += [samples,np.zeros(round(rate*pause),dtype=np.float32)]
        position = step["end"]
    audio=np.concatenate(segments)
    data["duration"]=round(len(audio)/rate,3)
    data["timingsVerified"]=True
    data["voiceEngine"]="Kokoro 82M ONNX · voce italiana im_nicola"
    data["speechSpeed"]=speed
    master=temporary/"master.wav"
    sf.write(str(master),audio,rate)
    data_path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    vtt="WEBVTT\n\n"
    srt=""
    for i,step in enumerate(data["steps"]):
        subtitle='\n'.join(textwrap.wrap(step["text"],width=66,break_long_words=False))
        vtt+=f"{i+1}\n{stamp(step['start'])} --> {stamp(step['end'])}\n{subtitle}\n\n"
        srt+=f"{i+1}\n{stamp(step['start'],',')} --> {stamp(step['end'],',')}\n{subtitle}\n\n"
    (MEDIA/"qkv.vtt").write_text(vtt,encoding="utf-8")
    (MEDIA/"qkv.srt").write_text(srt,encoding="utf-8")
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(master),"-c:a","libmp3lame","-b:a","96k",str(MEDIA/"qkv-nicola.mp3")],check=True)
    frames=[]
    for i,step in enumerate(data["steps"]):
        svg=scene_svg(step,i,len(data["steps"]))
        ET.fromstring(svg)
        if i==0:
            (MEDIA/"qkv-poster.svg").write_text(svg,encoding="utf-8")
        png=resvg_py.svg_to_bytes(svg_string=svg,font_files=["C:/Windows/Fonts/segoeui.ttf","C:/Windows/Fonts/segoeuib.ttf","C:/Windows/Fonts/consola.ttf"],skip_system_fonts=True)
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
    # SVG produces the static geometry; small dots trace attention paths over time.
    command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r",str(fps),"-i","pipe:0","-i",str(master),"-map","0:v","-map","1:a","-c:v","libx264","-preset","fast","-crf","24","-pix_fmt","yuv420p","-c:a","aac","-b:a","96k","-movflags","+faststart","-shortest",str(MEDIA/"qkv.mp4")]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for frame_number in range(math.ceil(data["duration"]*fps)):
            t=frame_number/fps
            i=next((j for j,step in enumerate(data["steps"]) if t<step["end"]),len(data["steps"])-1)
            step=data["steps"][i]
            frame=frames[i].copy()
            draw=ImageDraw.Draw(frame)
            local=max(0,t-step["start"])
            phase=(local%2.4)/2.4
            if step["stage"]=="query":
                x=255+40*phase;y=222
            elif step["stage"] in {"key","value","read"}:
                x=(1145-25*phase);y=203+51*int((local//2.4)%4)
            elif step["stage"]=="weights":
                target_y=203+51*int((local//2.4)%4)
                x=1048-313*phase;y=target_y+(358-target_y)*(3*phase**2-2*phase**3)
            elif step["stage"] in {"scores","output"}:
                target_y=203+51*int((local//2.4)%4)
                x=400+519*phase;y=222+(target_y-222)*(3*phase**2-2*phase**3)
            else:
                x=640;y=265
            draw.ellipse((x-7,y-7,x+7,y+7),fill=ORANGE)
            draw.rectangle((0,710,1280*min(1,t/data["duration"]),720),fill=INDIGO)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait()!=0:
            raise RuntimeError("ffmpeg video encoding failed")
    finally:
        if process.poll() is None:
            process.kill()
    print(json.dumps({"duration":data["duration"],"video":str(MEDIA/"qkv.mp4"),"bytes":(MEDIA/"qkv.mp4").stat().st_size}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--speed",type=float,default=.86)
    parser.add_argument("--fps",type=int,default=20)
    parser.add_argument("--reuse-audio",action="store_true")
    args=parser.parse_args()
    build(args.runtime,args.speed,args.fps,args.reuse_audio)


if __name__=="__main__":
    main()
