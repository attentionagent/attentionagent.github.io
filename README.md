# Neuroevolution of Self-Interpretable Agents

This repo contains the videos for our GECCO2020 [paper](https://arxiv.org/abs/2003.08165).

### If all you want to do is view the page locally

Run `python -m http.server` to serve on the base directory to view `index.html` in a local browser.

### Article

`draft.md` - main text of the article, in markdown.

`draft_appendix.md` - appendix, in markdown.

`draft_bib.html` - the citations.

`draft_header.html` - start of the document

`index.html` - generated, don't edit this file.

### Instructions to Build and Test
```bash
git clone https://github.com/attentionagent/attentionagent.github.io.git
cd attentionagent.github.io
npm install
```

Modify text by editing `draft.md` -- this is where all of the content exists.

Appendix content goes in `draft_appendix.md`. Add bib entries to `draft_bib.html`.

Run `./bin/make` to build document into `index.html` (which are identical).
Run `python -m http.server` to serve on the base directory to view `index.html` in a local browser for debugging.

To watch all markdown files for changes and then compile them, you can run the following
```
brew install fswatch
./bin/watch
```

### Citation

This work was presented at <a href="https://gecco-2020.sigevo.org/index.html/HomePage" target="_blank">GECCO 2020</a> as a full paper.

For attribution in academic contexts, please cite this work as

BibTeX citation
```
@inproceedings{attentionagent2020,
  author    = {Yujin Tang and Duong Nguyen and David Ha},
  title     = {Neuroevolution of Self-Interpretable Agents},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  url       = {https://attentionagent.github.io},
  note      = "\url{https://attentionagent.github.io}",
  year      = {2020}
}
```
