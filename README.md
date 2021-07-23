# 📝 simple parallel line fitting using RANSAC Linear Least Squares approximation
***

- fitting of 2 parallel lines: see details about the formulation in reference [2]

<p align="center">
  <img width="480" height="360" src="./docs/images/parallel_line.png">
</p>

- fitting of 2 parallel parabolas: not the official text-book definition, but here 2 parallel parabolas p<sub>0</sub>, p<sub>1</sub> are defined by: ax<sup>2</sup>+bx+c<sub>0</sub> (p<sub>0</sub>); ax<sup>2</sup>+bx+c<sub>1</sub> (p<sub>1</sub>). Then the parallel parabola fitting can be formulated by the following least squares problem:

<p align="center">
  <img src="./docs/images/parallel_parabola_ls_eq.gif">
</p>

<p align="center">
  <img width="480" height="360" src="./docs/images/parallel_parabola.png">
</p>

## :tada: TODO
***

- [x] fitting of two parallel lines, demo code
- [x] fitting of two parallel parabola, demo code
- [x] add julia code (just to practice julia)

## 🎛  Dependencies
***

- python

```bash
python -m pip install -r requirements.txt
```

- julia

```bash
pkg> activate .
pkg> instantiate
```

## :running: How to Run ##
***

- python

    + demo for fitting of two parallel lines

    ```bash
    python demo_parallel_line_fitting.py
    ```

    + demo for fitting of two parabolas

    ```bash
    python demo_parallel_parabola_fitting.py
    ```

- julia

    + demo for fitting of two parallel lines

    ```bash
    julia --project=. demo_parallel_line_fitting.jl
    ```

    + demo for fitting of two parallel parabolas

    ```bash
    julia --project=. demo_parallel_parabola_fitting.jl
    ```

**You can also call the demo scripts from julia REPL with include syntax. This will reduce the latency for code loading and compilation for the second run on REPL. However, you will always suffer from loading and compilation latency if running directly from command line.**

## :gem: References ##
***

- [1] [basic theory about least-squares data fitting](https://courses.grainger.illinois.edu/cs357/sp2021/notes/ref-17-least-squares.html)
- [2] [basic theory about parallel line fitting](http://people.inf.ethz.ch/arbenz/MatlabKurs/node86.html)
- [3] [a detailed blog about RANSAC 3D line fitting without SVD-based least squares method](https://zalo.github.io/blog/line-fitting/)
