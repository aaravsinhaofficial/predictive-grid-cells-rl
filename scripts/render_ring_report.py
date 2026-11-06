"""Render the finished mechanistic report and methods as a readable PDF.

Requires reportlab, markdown, beautifulsoup4 and pillow. Source Markdown and
figure files remain the canonical editable research record.
"""
import argparse
from html import escape
from pathlib import Path
import re

from bs4 import BeautifulSoup, NavigableString
import markdown
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                               Table, TableStyle, KeepTogether, PageBreak)
import matplotlib

ROOT=Path(__file__).resolve().parents[1]
WEB="https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/blob/main/"
WIDTH=6.75*inch


def clean(text):
    return text.translate(str.maketrans({c:"-" for c in "‐‑‒–—−"})).replace("\u00a0"," ")


def link_target(target,source):
    if target.startswith(("http://","https://","mailto:")):
        return target
    path,sep,fragment=target.partition("#")
    resolved=(source.parent/path).resolve()
    return WEB+str(resolved.relative_to(ROOT))+(sep+fragment if sep else "")


def inline(node,source):
    if isinstance(node,NavigableString):
        return escape(clean(str(node)))
    content="".join(inline(x,source) for x in node.children)
    if node.name in ("strong","b"):return "<b>"+content+"</b>"
    if node.name in ("em","i"):return "<i>"+content+"</i>"
    if node.name=="code":return '<font name="Mono" size="8.3">'+content+'</font>'
    if node.name=="a":return '<link href="'+escape(link_target(node.get("href",""),source),quote=True)+'" color="#226294">'+content+'</link>'
    if node.name=="br":return "<br/>"
    return content


def setup():
    folder=Path(matplotlib.get_data_path())/"fonts/ttf"
    for name,file in (("Body","DejaVuSerif.ttf"),("BodyBold","DejaVuSerif-Bold.ttf"),
                      ("BodyItalic","DejaVuSerif-Italic.ttf"),("BodyBoldItalic","DejaVuSerif-BoldItalic.ttf"),
                      ("Sans","DejaVuSans.ttf"),("SansBold","DejaVuSans-Bold.ttf"),
                      ("SansItalic","DejaVuSans-Oblique.ttf"),("Mono","DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name,str(folder/file)))
    pdfmetrics.registerFontFamily("Body",normal="Body",bold="BodyBold",italic="BodyItalic",boldItalic="BodyBoldItalic")
    pdfmetrics.registerFontFamily("Sans",normal="Sans",bold="SansBold",italic="SansItalic",boldItalic="SansBold")
    styles={
        "body":ParagraphStyle("body",fontName="Body",fontSize=10,leading=14,spaceAfter=8,allowWidows=0,allowOrphans=0),
        "title":ParagraphStyle("title",fontName="SansBold",fontSize=23,leading=28,spaceAfter=16,keepWithNext=True),
        "h2":ParagraphStyle("h2",fontName="SansBold",fontSize=13,leading=17,spaceBefore=14,spaceAfter=7,keepWithNext=True),
        "h3":ParagraphStyle("h3",fontName="SansBold",fontSize=10.5,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True),
        "caption":ParagraphStyle("caption",fontName="Sans",fontSize=8,leading=10.8,spaceAfter=11),
        "cell":ParagraphStyle("cell",fontName="Sans",fontSize=8,leading=10.5,spaceAfter=0),
        "code":ParagraphStyle("code",fontName="Mono",fontSize=7.5,leading=10,spaceBefore=4,spaceAfter=8,
                              backColor=colors.HexColor("#f2f4f5"),borderPadding=7),
        "bullet":ParagraphStyle("bullet",fontName="Body",fontSize=9.7,leading=13.5,leftIndent=12,firstLineIndent=-9,spaceAfter=5),
    }
    return styles


def blocks(source,styles,allow_draft=False):
    text=source.read_text()
    if not allow_draft and any(flag in text for flag in ("will be inserted","Working report —","confirmation is ongoing")):
        raise ValueError(f"Refusing to render unfinished report: {source}")
    text=re.sub(r"<!--.*?-->","",text,flags=re.S)
    html=markdown.markdown(text,extensions=["tables","fenced_code"])
    soup=BeautifulSoup(html,"html.parser")
    nodes=[x for x in soup.contents if not isinstance(x,NavigableString) or str(x).strip()]
    story=[];i=0
    while i<len(nodes):
        node=nodes[i];i+=1
        name=getattr(node,"name",None)
        if name in ("h1","h2","h3","h4"):
            style="title" if name=="h1" else "h2" if name=="h2" else "h3"
            story.append(Paragraph(inline(node,source),styles[style]))
        elif name=="p" and node.find("img"):
            image=node.find("img");path=(source.parent/image["src"]).resolve()
            with PILImage.open(path) as img:w,h=img.size
            width=WIDTH; height=width*h/w
            if height>5.5*inch:height=5.5*inch;width=height*w/h
            group=[Image(str(path),width=width,height=height),Spacer(1,5)]
            if i<len(nodes) and getattr(nodes[i],"name",None)=="p" and nodes[i].find("em"):
                group.append(Paragraph(inline(nodes[i],source),styles["caption"]));i+=1
            story.append(KeepTogether(group))
        elif name=="p":
            style=styles["body"]
            following = nodes[i] if i < len(nodes) else None
            codes = following.find_all("code") if getattr(following,"name",None)=="p" else []
            equation_next = bool(codes) and sum(len(c.get_text()) for c in codes) > .75 * len(following.get_text())
            if node.get_text().rstrip().endswith((":"," by")) or equation_next:
                style=ParagraphStyle("lead",parent=style,keepWithNext=True)
            story.append(Paragraph(inline(node,source),style))
        elif name in ("ul","ol"):
            for j,item in enumerate(node.find_all("li",recursive=False),1):
                prefix=f"{j}. " if name=="ol" else "• "
                story.append(Paragraph(prefix+inline(item,source),styles["bullet"]))
            story.append(Spacer(1,4))
        elif name=="table":
            rows=[]
            for tr in node.find_all("tr"):
                row=[]
                for cell in tr.find_all(["td","th"],recursive=False):
                    value=inline(cell,source)
                    if cell.name=="th":value="<b>"+value+"</b>"
                    row.append(Paragraph(value,styles["cell"]))
                rows.append(row)
            n=len(rows[0]);ratios=([.38,.62] if n==2 else [.40]+[.60/(n-1)]*(n-1))
            table=Table(rows,colWidths=[WIDTH*r for r in ratios],repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf0f4")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),
                ("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),
                ("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#afbdc7")),
                ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#dce2e6"))]))
            story.extend([table,Spacer(1,10)])
        elif name=="pre":
            # Paragraph wrapping prevents long reproduction commands from clipping.
            value=escape(clean(node.get_text())).replace("\n","<br/>")
            story.append(Paragraph(value,styles["code"]))
    return story


def page(canvas,doc):
    canvas.saveState();canvas.setFont("Sans",7.5);canvas.setFillColor(colors.HexColor("#65717a"))
    canvas.drawString(.875*inch,.47*inch,"Predictive spatial codes and recurrent memory | Computational research report")
    canvas.drawRightString(7.625*inch,.47*inch,str(doc.page))
    canvas.restoreState()


def main(args):
    styles=setup();story=[]
    sources=[ROOT/"mechanisms/predictive-ring-attractor.md"]
    if args.methods:sources.append(ROOT/"mechanisms/predictive-ring-methods.md")
    if args.derivation:sources.append(ROOT/"docs/research/predictive_ring_derivation.md")
    for i,source in enumerate(sources):
        if i:story.append(PageBreak())
        story.extend(blocks(source,styles,args.allow_draft))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(args.out),pagesize=(8.5*inch,11*inch),
        leftMargin=.875*inch,rightMargin=.875*inch,topMargin=.7*inch,bottomMargin=.75*inch,
        title="Predictive spatial codes anchor recurrent direction memory",
        author="Predictive grid-cell research project",subject="Mechanistic models, trained recurrent networks and causal interventions")
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(args.out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"output/pdf/predictive_ring_mechanism.pdf")
    parser.add_argument("--methods",action="store_true")
    parser.add_argument("--derivation",action="store_true")
    parser.add_argument("--allow-draft",action="store_true",help="Permit visibly labeled draft status for local layout review")
    main(parser.parse_args())
