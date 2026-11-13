"""Reproduce the current manuscript figures from archived data, without training.

Examples: python scripts/reproduce_paper.py --list
          python scripts/reproduce_paper.py --figure 10
          python scripts/reproduce_paper.py --figure all
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'assets/paper_figures/manifest.json'


def manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def build_schematics(output):
    """Compile editable TikZ sources; never install software or modify sources."""
    latex = shutil.which('pdflatex')
    if not latex:
        raise RuntimeError('Install a TeX distribution with pdflatex, TikZ/PGFPlots and standalone, '
                           'or omit --build-schematics to use the verified PDFs.')
    from pypdf import PdfReader, PdfWriter
    with tempfile.TemporaryDirectory(prefix='core-pivot-figures-') as tmp:
        work = Path(tmp)
        shutil.copytree(ROOT/'assets/paper_sources', work, dirs_exist_ok=True)
        for source in ['render_insets_no_titles.tex', 'search_panel_column.tex']:
            result = subprocess.run([latex, '-interaction=nonstopmode', '-halt-on-error', source],
                                    cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:] + result.stderr[-1000:])
        # Exact vector-panel crops used for the manuscript; no rasterisation.
        for stem, box in [('workflow',(1,153,457,306)), ('tracking_task',(242,17,447,132))]:
            page = PdfReader(work/'render_insets_no_titles.pdf').pages[0]
            page.mediabox.lower_left = box[:2]
            page.mediabox.upper_right = box[2:]
            page.cropbox = page.mediabox
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(output/f'{stem}.pdf')
        shutil.copyfile(work/'search_panel_column.pdf', output/'action_search.pdf')


def reproduce(selected, output, build=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = manifest()
    # Validate the static assets before any copies; numeric functions verify their archives.
    for name, digest in archive['schematic_sha256'].items():
        if hashlib.sha256((ROOT/'assets/paper_figures'/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Schematic hash mismatch: {name}')
    from scripts import plot_paper as plots
    plots.plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
                              'font.size':7, 'axes.labelsize':7, 'svg.fonttype':'none', 'pdf.fonttype':42})
    data = plots.verify()
    done = []
    if build and set(selected) & {1,2,3}:
        build_schematics(output)
    jobs = {4: lambda: plots.dataset(output), 5: lambda: plots.dataset(output),
            6: lambda: plots.learning(data,output), 7: lambda: plots.robustness(data,output),
            8: lambda: plots.ablations(data,output), 10: lambda: plots.test_violins(data,output),
            11: lambda: plots.learning(data,output,fullrange=True)}
    dataset_done = False
    for item in archive['figures']:
        number, stem = item['id'], item['stem']
        if number not in selected:
            continue
        if number <= 3:
            if not build:
                shutil.copyfile(ROOT/'assets/paper_figures'/f'{stem}.pdf', output/f'{stem}.pdf')
        elif number in jobs:
            if number not in (4,5) or not dataset_done:
                jobs[number]()
            if number in (4,5): dataset_done = True
        else:
            from scripts.plot_supplement import reproduce_figure
            reproduce_figure(number, output)
        path = output/f'{stem}.pdf'
        if not path.is_file():
            raise RuntimeError(f'Figure {number} was not produced: {path}')
        done.append(dict(figure=number, file=path.name, caption=item['caption'],
                         mode='compiled TikZ' if number<=3 and build else item['kind'],
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f'Figure {number}: {path.name}')
    if set(selected) == set(range(1,20)):
        from scripts.export_tables import export_tables
        export_tables(data, output)
        from scripts.plot_supplement import export_trajectory_table
        export_trajectory_table(output)
        # Only exact obsolete generated names, after replacements have succeeded.
        for stem in ['benchmark_dataset','original_test_violins','original_test_summary','robustness','four_ablations_fullrange']:
            for ext in ('svg','pdf','png','json'):
                old = output/f'{stem}.{ext}'
                if old.is_file() and not old.is_symlink():
                    old.unlink()
                    print(f'Removed obsolete generated plot: {old.name}')
    (output/'figure_index.json').write_text(json.dumps({'paper_revision':archive['paper_revision'],
        'figures':done, 'training_performed':False},indent=2)+'\n',encoding='utf-8')
    sections = []
    for item in done:
        pdf = html.escape(item['file'])
        png = Path(item['file']).with_suffix('.png').name
        preview = (f'<img loading="lazy" src="{html.escape(png)}" alt="Figure {item["figure"]}">'
                   if (output/png).is_file() else '')
        sections.append(f'<section><h2>Figure {item["figure"]}</h2><p>{html.escape(item["caption"])}</p>'
                        f'<a href="{pdf}">Open PDF</a>{preview}</section>')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>CORE-PIVOT paper figures</title><style>body{font:16px system-ui;max-width:1000px;'
        'margin:40px auto;padding:0 20px}section{margin:40px 0}img{display:block;max-width:100%;'
        'height:auto;margin-top:12px}</style><h1>CORE-PIVOT paper figures</h1>'
        '<p>Archived results; no training or new evaluations. '+html.escape(archive['paper_revision'])+'</p>'
        +''.join(sections)+'</html>',encoding='utf-8')
    print(f'Completed {len(done)} figures. No training, evaluation or checkpoint download was performed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true', help='Show current figure numbers without loading scientific libraries')
    p.add_argument('--figure', default='all', help='all or one figure number (1–19)')
    p.add_argument('--output', type=Path, default=ROOT/'figures/reproduced')
    p.add_argument('--build-schematics', action='store_true', help='Recompile TikZ diagrams with an installed pdflatex')
    args = p.parse_args()
    if args.list:
        for f in manifest()['figures']: print(f"{f['id']:2}: {f['caption']} [{f['kind']}]")
        return
    try:
        selected = list(range(1,20)) if args.figure=='all' else [int(args.figure)]
        if not all(1<=n<=19 for n in selected): raise ValueError()
    except ValueError:
        p.error('--figure must be all or an integer from 1 to 19')
    reproduce(selected,args.output,args.build_schematics)


if __name__ == '__main__':
    main()
