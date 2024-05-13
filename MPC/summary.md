## Optimization Goal

$$
\begin{aligned}
&\mathrm{min}\quad J = \sum_{i = 0}^{N}||x_i - x_{ref,i}||_Q^2 + ||u_i||_R^2\\
&s.t. \quad \dot{x} = f(x(t), u(t))\\
\end{aligned}
$$

$$
\cfrac{\mathrm{d}x}{\mathrm{d}s} = \cfrac{\mathrm{d}x}{\mathrm{d}t}\cdot \cfrac{\mathrm{d}t}{\mathrm{d}s} = \dot{x}\cdot \mathrm{SF}
$$


状态转移约束： 
frenet域：
$$
x(k+1) = x(k) + \cfrac{\mathrm{d}x}{\mathrm{d}s}\cdot\Delta s = x(k) + \Delta s\cdot\dot{x}\cdot\mathrm{SF}
$$

时域：
$$
x(k+1) = x(k) + \cfrac{\mathrm{d}x}{\mathrm{d}t}\cdot\Delta t = x(k) + \dot{x}\Delta t
$$

模型约束：
* 控制变量范围约束
* equ8,10,11,12,14
<!-- * (连续性约束/首尾约束) -->