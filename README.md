### ThinkBayes in Julia

This repository supports my implementation of notebooks that implement the notebooks of Allen Downey's [ThinkBayes2](http://allendowney.github.io/ThinkBayes2/index.html) in Julia. Why Julia? 
1. I wanted to learn Julia
2. I felt like I know python so well, that working through the notebooks in python, I'd gloss over important concepts. To reimplement in a different (while obviously similar) language would force me to understand at a deeper level. YMMV.

Having gone down this path, I'm very exited about Julia and it's ecosystem. Pluto is an amazing (mostly) upgrade from jupyter notebooks. A lot of the infrastructure in python that is built around numpy is essentially built in to the core of Julia.

There are some rough spots. DataFrames in Julia aren't as powerful as Pandas, but this is apparently by design. However, it does make for some impedence mismatching when trying to replicate python code that makes heavy use of Pandas.

Notes:
1. As of now this repository only supports the first eleven chapters of Think Bayes 2. I'll update this notice as I make more progress.
2. The first 9 notebook chapters are jupyter notebooks. The remaining chapters are pluto notebooks. My intention is to redo the jupyter notebooks to pluto, but....
