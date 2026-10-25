#!/usr/bin/env python3
"""Section 5: vector plots from frozen results, without training or model calls.

Run with the bundled Codex Python (reportlab is already provided). The source
JSON hashes, exact plotted values, and geometric text bounds are saved alongside
the figures. No inferential test is recomputed and no trajectory is excluded.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
import subprocess

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WIDTH = 420
COLORS = {'A': '#447DA6', 'P25': '#D78039', 'C': '#447DA6',
          'P': '#D78039', 'D': '#82629F'}
INK = '#333333'
GRID = '#E3E5E8'
BACKGROUND = '#FFFFFF'
GRAY = '#AFB3B8'
SOURCES = {}
VALUES = {'strength': {}, 'scale': {}, 'horizon': {}, 'external': {}}
BOUNDS = []

pdfmetrics.registerFont(TTFont('ZH', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=1))


def read(relative):
    path = ROOT / relative
    SOURCES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


STRENGTH = read('experiments/exp_038_protection_strength/outputs/analysis.json')
SCALE = read('experiments/exp_040_scale_generalization_v2/final_results/cloud4090_20261021/outputs/analysis.json')
HORIZON = read('experiments/exp_042_seed_and_horizon/final_results/cloud4090_20261021/outputs/analysis.json')
EXTERNAL = read('experiments/exp_043_kuzushiji49_generalization/multiseed_10M_20261022/summary/combined_analysis.json')


class Figure:
    def __init__(self, name, language, height):
        self.language = language
        self.path = HERE / f'{name}_{language}.pdf'
        self.height = height
        self.c = canvas.Canvas(str(self.path), pagesize=(WIDTH, height), pageCompression=1)
        self.c.setTitle(name)
        self.c.setAuthor('ICL dynamics project')
        self.bounds = []

    def tr(self, en, zh):
        return zh if self.language == 'zh' else en

    def text(self, x, y, text, size=8.5, align='left', color=INK, bold=False, angle=0):
        text = str(text)
        font = 'ZH' if re.search(r'[\u3400-\u9fff]', text) else ('Helvetica-Bold' if bold else 'Helvetica')
        width = stringWidth(text, font, size)
        offset = {'left': 0, 'center': -width/2, 'right': -width}[align]
        self.c.saveState()
        self.c.translate(x, y)
        self.c.rotate(angle)
        self.c.setFillColor(HexColor(color))
        self.c.setFont(font, size)
        self.c.drawString(offset, 0, text)
        self.c.restoreState()
        if not angle:
            assert x + offset >= -0.5 and x + offset + width <= WIDTH + 0.5, (self.path, text, x, width)
            assert 0 <= y <= self.height, (self.path, text, y)
        self.bounds.append({'text': text, 'x': x, 'y': y, 'width': width,
                            'size': size, 'angle': angle})

    def line(self, x1, y1, x2, y2, color=INK, width=.65, dash=None):
        self.c.saveState()
        self.c.setStrokeColor(HexColor(color))
        self.c.setLineWidth(width)
        if dash:
            self.c.setDash(*dash)
        self.c.line(x1, y1, x2, y2)
        self.c.restoreState()

    def rect(self, x, y, width, height, color):
        self.c.saveState()
        self.c.setFillColor(HexColor(color))
        self.c.rect(x, y, width, height, fill=1, stroke=0)
        self.c.restoreState()

    def marker(self, x, y, color, shape='circle', radius=2.8, filled=True):
        self.c.saveState()
        self.c.setFillColor(HexColor(color if filled else '#FFFFFF'))
        self.c.setStrokeColor(HexColor(color))
        self.c.setLineWidth(.9)
        if shape == 'square':
            self.c.rect(x-radius, y-radius, radius*2, radius*2, fill=1, stroke=1)
        elif shape == 'triangle':
            p = self.c.beginPath()
            p.moveTo(x, y+radius*1.2)
            p.lineTo(x-radius*1.15, y-radius)
            p.lineTo(x+radius*1.15, y-radius)
            p.close()
            self.c.drawPath(p, fill=1, stroke=1)
        else:
            self.c.circle(x, y, radius, fill=1, stroke=1)
        self.c.restoreState()

    def finish(self):
        self.c.showPage()
        self.c.save()
        BOUNDS.append({'file': self.path.name, 'labels': self.bounds})
        poppler = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdftoppm'
        subprocess.run([str(poppler), '-singlefile', '-scale-to', '1500', '-png',
                        str(self.path), str(self.path.with_suffix(''))],
                       check=True, capture_output=True)


class Axes:
    def __init__(self, fig, x, y, w, h, xlim, ylim, xticks, yticks,
                 xlabel='', ylabel='', title='', xlabels=None):
        self.f, self.x, self.y, self.w, self.h = fig, x, y, w, h
        self.xlim, self.ylim = xlim, ylim
        fig.rect(x, y, w, h, BACKGROUND)
        for tick in yticks:
            yy = self.Y(tick)
            fig.line(x, yy, x+w, yy, GRID, .45)
            fig.text(x-5, yy-2.8, f'{tick:g}', 8, 'right')
        fig.line(x, y, x+w, y, '#7F858B', .6)
        fig.line(x, y, x, y+h, '#7F858B', .6)
        for i, tick in enumerate(xticks):
            label = xlabels[i] if xlabels else f'{tick:g}'
            fig.line(self.X(tick), y, self.X(tick), y-2.5, '#7F858B', .55)
            fig.text(self.X(tick), y-13, label, 8, 'center')
        if xlabel:
            fig.text(x+w/2, y-29, xlabel, 8.8, 'center')
        if ylabel:
            fig.text(x-32, y+h/2, ylabel, 8.8, 'center', angle=90)
        fig.text(x+w/2, y+h+11, title, 9.4, 'center', bold=False)

    def X(self, value):
        return self.x+(value-self.xlim[0])/(self.xlim[1]-self.xlim[0])*self.w

    def Y(self, value):
        return self.y+(value-self.ylim[0])/(self.ylim[1]-self.ylim[0])*self.h

    def points(self, xs, ys, color, shape='circle', connect=True, width=1.4,
               radius=2.8, dash=None):
        assert len(xs) == len(ys)
        assert all(self.xlim[0] <= v <= self.xlim[1] for v in xs)
        assert all(self.ylim[0] <= v <= self.ylim[1] for v in ys)
        if connect:
            for a, b, c, d in zip(xs, ys, xs[1:], ys[1:]):
                self.f.line(self.X(a), self.Y(b), self.X(c), self.Y(d), color, width, dash)
        if radius:
            for a, b in zip(xs, ys):
                self.f.marker(self.X(a), self.Y(b), color, shape, radius)

    def interval(self, x, lo, hi, color, width=.8, caps=2):
        assert self.ylim[0] <= lo <= hi <= self.ylim[1], (lo, hi, self.ylim)
        xx = self.X(x)
        self.f.line(xx, self.Y(lo), xx, self.Y(hi), color, width)
        for yy in [self.Y(lo), self.Y(hi)]:
            self.f.line(xx-caps, yy, xx+caps, yy, color, width)


def legend(fig, entries, y, width=360):
    lengths = [20 + stringWidth(label, 'ZH' if re.search(r'[\u3400-\u9fff]', label) else 'Helvetica', 8.1)
               for label, _, _ in entries]
    gap = (width-sum(lengths))/(len(entries)-1) if len(entries)>1 else 0
    assert gap >= 0
    x = (WIDTH-width)/2
    for (label, color, shape), length in zip(entries, lengths):
        fig.line(x, y, x+15, y, color, 1.6)
        fig.marker(x+7.5, y, color, shape, 2.5)
        fig.text(x+20, y-2.8, label, 8.1)
        x += length+gap


def strength_plot(language):
    f = Figure('fig_s5_1_strength', language, 174)
    rhos = [.10, .25, .50, .75]
    settings = [('icl', (80, 95), [80, 85, 90, 95], '(a) Standard ICL', '(a) 标准 ICL'),
                ('iwl_copy_avail', (82, 91), [82, 85, 88, 91], '(b) CIWL', '(b) CIWL')]
    for i, (task, ylim, ticks, title, title_zh) in enumerate(settings):
        ax = Axes(f, 40+i*210, 49, 157, 96, (.06, .79), ylim, rhos, ticks,
                  f.tr('Protection coefficient', '保护系数'),
                  f.tr('Accuracy (%)', '准确率（%）'), f.tr(title, title_zh),
                  ['0.10','0.25','0.50','0.75'])
        for method, shape in [('C','square'),('D','triangle'),('P','circle')]:
            rows = [STRENGTH['descriptive'][method+str(round(r*100))][task]['acc'] for r in rhos]
            color = COLORS[method]
            for rho, row in zip(rhos, rows):
                ax.interval(rho, row['ci95_lower']*100, row['ci95_upper']*100, color)
            ax.points(rhos, [r['mean']*100 for r in rows], color, shape)
            VALUES['strength'][f'{method}/{task}'] = [{'rho': r, 'mean_pct': a['mean']*100,
                  'ci95_pct': [a['ci95_lower']*100, a['ci95_upper']*100]} for r,a in zip(rhos,rows)]
    legend(f, [(f.tr('Task-loss C','任务损失 C'),COLORS['C'],'square'),
               (f.tr('Combined D','组合约束 D'),COLORS['D'],'triangle'),
               (f.tr('Pathway P','通路保护 P'),COLORS['P'],'circle')], 9)
    f.finish()


def scale_plot(language):
    f = Figure('fig_s5_2_scale', language, 174)
    for i, (key, name) in enumerate([('base','2L/64'),('width','2L/128'),('depth','4L/128')]):
        a = SCALE['architectures'][key]
        assert set(a['seeds']) == {'s13','s14','s15','s16','s17'}
        ax = Axes(f, 42+i*137, 49, 94, 96, (-.25,1.25),(40,100), [0,1],[40,60,80,100],
                  '', f.tr('ICL accuracy (%)','ICL 准确率（%）') if i==0 else '',
                  f'({chr(97+i)}) {name}', ['A','P25'])
        vals = []
        for j, seed in enumerate(sorted(a['seeds'])):
            row = a['seeds'][seed]
            av,pv = [row['means'][arm]['icl']['acc']['mean']*100 for arm in ['A','P25']]
            jitter = (j-2)*.028
            f.line(ax.X(jitter),ax.Y(av),ax.X(1+jitter),ax.Y(pv),GRAY,.65)
            f.marker(ax.X(jitter),ax.Y(av),COLORS['A'],'square',2.8)
            f.marker(ax.X(1+jitter),ax.Y(pv),COLORS['P25'],'circle',2.8)
            vals.append({'seed':seed,'A_pct':av,'P25_pct':pv,
                         'confirmation_passed':row['mechanism']['gate_passed']})
        for pos,arm in [(0,'A'),(1,'P25')]:
            avg = a['arm_means'][arm]['icl']['acc']['mean']*100
            f.line(ax.X(pos-.14),ax.Y(avg),ax.X(pos+.14),ax.Y(avg),INK,1.65)
        VALUES['scale'][key] = vals
    legend(f,[(f.tr('Ordinary A','正常训练 A'),COLORS['A'],'square'),
              (f.tr('Pathway P25','通路保护 P25'),COLORS['P25'],'circle')],14,290)
    f.finish()


def horizon_plot(language):
    f = Figure('fig_s5_3_horizon',language,182)
    budgets = [10000000,20000000,30000000,40000000]
    cohort = ['s18','s19','s20']
    tasks = [('icl/acc','(a) Standard ICL','(a) 标准 ICL',(30,100),[40,60,80,100]),
             ('iwl_copy_avail/acc','(b) CIWL','(b) CIWL',(60,100),[60,80,100]),
             ('train_eval/acc','(c) Original task','(c) 原分类任务',(75,100),[75,85,95,100])]
    for i,(task,title,title_zh,ylim,yticks) in enumerate(tasks):
        ax = Axes(f,42+i*137,52,94,96,(9,41),ylim,[10,20,30,40],yticks,
                 f.tr('Sequences (M)','序列数（百万）'),f.tr('Accuracy (%)','准确率（%）') if i==0 else '',
                 f.tr(title,title_zh))
        for arm,shape in [('A','square'),('P25','circle')]:
            trajectories=[]
            for seed in cohort:
                vals=[]
                for b in budgets:
                    rows = [v for v in HORIZON['per_seed'] if v['budget']==b and v['seed']==seed and v['arm']==arm]
                    assert len(rows)==1,(seed,arm,b)
                    vals.append(rows[0]['metrics'][task]*100)
                faint = '#AFC8DA' if arm == 'A' else '#E9C7AD'
                ax.points([10,20,30,40],vals,faint,shape,True,.6,0,[2,2])
                trajectories.append({'seed':seed,'accuracy_pct':vals})
            means=[statistics.mean(v['accuracy_pct'][j] for v in trajectories) for j in range(4)]
            ax.points([10,20,30,40],means,COLORS[arm],shape,True,1.7,2.5)
            VALUES['horizon'][f'{arm}/{task}']={'budgets':budgets,'seeds':trajectories,'means_pct':means}
    legend(f,[(f.tr('Ordinary A','正常训练 A'),COLORS['A'],'square'),
              (f.tr('Pathway P25','通路保护 P25'),COLORS['P25'],'circle')],9,290)
    f.finish()


def external_plot(language):
    f = Figure('fig_s5_4_external',language,180)
    for i,(key,name) in enumerate([('k49','Kuzushiji-49'),('cifar100','CIFAR-100')]):
        d=EXTERNAL['datasets'][key]
        assert len(d['per_seed'])==5
        ax=Axes(f,40+i*210,54,157,94,(17.6,22.4),(-5,25),[18,19,20,21,22],[-5,0,10,20],
                f.tr('Training seed','训练种子'),f.tr('P25 - A gain (pp)','P25 - A 增益（百分点）'),
                f'({chr(97+i)}) {name}', ['s18','s19','s20','s21','s22'])
        f.line(ax.x,ax.Y(0),ax.x+ax.w,ax.Y(0),'#73737B',.7,[3,2])
        for row in d['per_seed']:
            seed=int(row['seed'][1:]);gain=row['P25_minus_A']['acc']*100
            f.line(ax.X(seed),ax.Y(0),ax.X(seed),ax.Y(gain),COLORS['P25'],1.2)
            f.marker(ax.X(seed),ax.Y(gain),COLORS['P25'],'circle',3.2)
        stat=d['P25_minus_A']['acc'];mean=stat['mean']*100
        f.line(ax.x,ax.Y(mean),ax.x+ax.w,ax.Y(mean),'#88624A',1.0,[5,2])
        VALUES['external'][key]={'per_seed':[{'seed':r['seed'],'gain_pp':r['P25_minus_A']['acc']*100} for r in d['per_seed']],
                   'mean_gain_pp':mean,'ci95_pp':[stat['ci95']['lower']*100,stat['ci95']['upper']*100],
                   'bonferroni_family95_pp':[stat['bonferroni_two_datasets_ci97_5']['lower']*100,
                                             stat['bonferroni_two_datasets_ci97_5']['upper']*100]}
    f.text(WIDTH/2,9,f.tr('Points: paired seeds    Dashed line: five-seed mean',
                            '圆点：配对种子    虚线：五种子均值'),8.3,'center')
    f.finish()


if __name__=='__main__':
    for language in ['en','zh']:
        for plot in [strength_plot,scale_plot,horizon_plot,external_plot]:
            plot(language)
    receipt={'source_sha256':SOURCES,'values':VALUES,
             'evaluation':'Full-label accuracy only; saved intervals are not recomputed.',
             'cohorts':{'scale':'all five s13-s17, including failed confirmation',
                        'horizon':'same prespecified s18-s20 at all four budgets',
                        'external':'all five s18-s22; negative CIFAR-100 seed retained'},
             'reference_style':{'palette':'User-requested white background; contrast-adjusted blue/orange/purple; no colored panel fill',
                'panels':'Strategy Coopetition (ICML 2025), Appendix Fig. 7; A-GEM (ICLR 2019), Fig. 3: aligned compact panels and shared legends',
                'longitudinal':'ICL Transience (NeurIPS 2023), Figs. 5-7: white panels and restrained grid',
                'canvas_width_pt':WIDTH,'target_manuscript_width':'0.82 textwidth, approximately 150 mm; >=8pt ticks'},
             'no_fitting_or_smoothing':True,'no_new_model_evaluation':True}
    (HERE/'plotted_values.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (HERE/'text_bounds.json').write_text(json.dumps(BOUNDS,ensure_ascii=False,indent=2)+'\n')
    print('Wrote eight vector PDFs; four plot types x two languages. Frozen values and source hashes recorded.')
