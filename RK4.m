function [t, y] = RK4(f, tspan, y0, h, c1, c2, c3, c4, c5)
    % tspan 是时间的起点和终点
    % y0 是方程初值
    % h 是步长
    t0 = tspan(1);
    tf = tspan(2);
    t = t0:h:tf;
    n = length(t);
    y = zeros(length(y0), n);
    a = zeros(n);
%     af = zeros(n);
    y(:, 1) = y0;
%     af(1) = af0;
    for i=1:n-1
        k1 = f(t(i), y(:, i), c1, c2, c3, c4, c5);
        k2 = f(t(i) + 0.5*h, y(:, i) + 0.5*h*k1, c1, c2, c3, c4, c5);
        k3 = f(t(i) + 0.5*h, y(:, i) + 0.5*h*k2, c1, c2, c3, c4, c5);
        k4 = f(t(i) + h, y(:, i) + h*k3, c1, c2, c3, c4, c5);
        y(:, i+1) = y(:, i) + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*h;
        %af(i+1) = c1*(y(2,i)+c2)./(y(1,i)+c3)+c4*(y(2,i)+0.209062)*(y(2,i)+0.209062)-0.063757245*af(i);
    end
    % title('y^,-t');

end
% 使用这个函数需要提供以下参数：
% 
% f：要解决的方程，是一个函数句柄，它接受一个t值和一个y值并返回当前的斜率。
% tspan：时间的起点和终点，是一个长度为2的向量。
% y0：方程的初始值，是一个列向量。
% tspan = [t0, tf], y0 = [y1(t0), y2(t0)], h为步长
% h：步长，即t的增量。
% t0 = 0;
% tf = 2;
% h = 0.01;
% y0 = [1; 0];
% [t, y] = RK4(@f, [t0 tf], y0, h);
% 
% plot(t, y(1, :));
% xlabel('t');
% ylabel('y');
