function dydt = f(t, y, c1, c2, c3, c4, c5)
    dydt =[y(2);c1*(y(2)+c2)/(y(1)+c4)-28.00798232*(y(2)+c3)*(y(2)+c3)-8.63406];%润滑力+伴流阻力+附加质量力
    %dydt = [y(2);c1*(y(2)+c2)/(y(1)+c3)];%仅润滑力
end
