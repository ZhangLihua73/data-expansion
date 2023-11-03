clc
clear
pathname='F:\ZLH\Basilisk\share\vertical\cell2\14bounce\';
divide=load([pathname,'divide_impact.mat']);
err2=0.00901;
err3=4.83626E-7;
c1=-0.00385;
c2=0.04099;
c3=-6.16294E-7;
c2_array=(c2-err2):1e-3:(c2+err2);
c3_array=(c3-err3):1e-8:(c3+err3);



tf=divide.test(1,1);%tf=0.036144000000000;

h=0.0000001;

%位移，速度
t0=divide.train(1,1);%t0=0.035392600000000;
y0 = [divide.train(1,3);divide.train(1,2)];

h_end=divide.test(1,3);
v_end=divide.test(1,2);
a_end=divide.test(1,4);
double a_test;
double c2;
double c3;
t_array=t0:h:tf;
%起始与结束时刻不变，确定合适的c1、c2误差值
err_a=zeros(length(c2_array),length(c3_array));
for i=1:length(c2_array)
    for j=1:length(c3_array)
        c2=c2_array(i);
        c3=c3_array(j);
        tspan = [t0, tf];
        [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3);
        for k=1:length(t)
            a(k)=c1*(y(2,k)+c2)./(y(1,k)+c3);
        end
        err_a(i,j)=abs((a_end-a(length(t)))/a_end);
    end
end
err_min_a=1;
for i=1:length(c2_array)
    for j=1:length(c3_array)
        if(err_a(i,j)<err_min_a)
            err_min_a=err_a(i,j);
            c2=c2_array(i);
            c3=c3_array(j);
        end
    end
end
%结束时刻不变，改变起始时刻，确定外推误差最小的起始时刻
err_t=zeros(length(divide.train(:,1)));
for k=1:length(divide.train(:,1))
    t0=divide.train(k,1);
    tspan = [t0, tf];
    y0 = [divide.train(k,3);divide.train(k,2)];
    [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3);
    for m=1:length(t)
        a(m)=c1*(y(2,m)+c2)./(y(1,m)+c3);
    end
    err_t(k)=abs((a_end-a(length(t)))/a_end);
end
err_min_t=1;
for k=1:length(divide.train(:,1))
    if(err_t(k)<err_min_t)
        err_min_t=err_t(k);
        t0=divide.train(k,1);
        y0 = [divide.train(k,3);divide.train(k,2)];
    end
end
tspan = [t0, tf];
[t,y] = RK4(@f, tspan, y0, h, c1, c2, c3);
%a_s,a_extend,error,t,h,v
a_extend=zeros(length(divide.test(:,1)),6);
%%%%%根据test对应时刻扩展
for i=1:length(divide.test(:,1))
    tf=divide.test(i,1);
    tspan = [t0, tf];
    [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3);
    for k=1:length(t)
        a_t(k)=c1*(y(2,k)+c2)./(y(1,k)+c3);
    end
    a_extend(i,1)=divide.test(i,4);
    a_extend(i,2)=a_t(length(t));
    a_extend(i,3)=abs((a_t(length(t))-divide.test(i,4))./divide.test(i,4));
    a_extend(i,4)=tf;
    a_extend(i,5)=y(1,length(t));
    a_extend(i,6)=y(2,length(t));
end
save ([pathname,'extend.mat'],'a_extend')
figure;
plot(t, y(2, :),'k-*',divide.test(:,1),divide.test(:,2),'r-')
xlabel('时间(s)','FontSize',15,'FontName','Times New Rome');
ylabel('速度(m/s)','FontSize',15,'FontName','Times New Rome');
legend('数据扩展','數值模拟','FontSize',15,'FontName','Times New Rome');
set(gca,'FontName','Times New Rome','FontSize',15);
figure;
plot(t, y(1,:),'ks',divide.test(:,1),divide.test(:,3),'r-')
xlabel('时间(s)','FontSize',15,'FontName','Times New Rome');
ylabel('距离(m)','FontSize',15,'FontName','Times New Rome');
legend('数据扩展','數值模拟','FontSize',15,'FontName','Times New Rome');
set(gca,'FontName','Times New Rome','FontSize',15);
figure;
plot(t, a_t,'ks',divide.test(:,1),divide.test(:,4),'rd')
xlabel('时间(s)','FontSize',15,'FontName','Times New Rome');
ylabel('加速度(m/s^2)','FontSize',15,'FontName','Times New Rome');
legend('数据扩展','数值模拟','FontSize',15,'FontName','Times New Rome');
set(gca,'FontName','Times New Rome','FontSize',15);
load([pathname,'extend.mat'])
figure;
plot(a_extend(:,4), a_extend(:,2),'ks',divide.test(:,1),divide.test(:,4),'r+')
xlabel('t(s)','FontSize',15,'FontName','Times New Rome');
ylabel('a(m/s^2)','FontSize',15,'FontName','Times New Rome');
legend('extend','simulation','FontSize',15,'FontName','Times New Rome');
set(gca,'FontName','Times New Rome','FontSize',15);
figure;
plot(a_extend(:,4), a_extend(:,3),'k-*')
xlabel('t(s)','FontSize',15,'FontName','Times New Rome');
ylabel('error','FontSize',15,'FontName','Times New Rome');
set(gca,'FontName','Times New Rome','FontSize',15);

%%%%%%%%%自由扩展，不用数值模拟数据进行测试
h=0.000001
tf=0.0362;
tspan = [t0, tf];
[t,y] = RK4(@f, tspan, y0, h, c1, c2, c3);
%a_s,a_extend,error,t,h,v
a_extend_free_impact=zeros(length(t),6);
%%%%%根据test对应时刻扩展
for i=1:length(t)
    %a_extend_free(i,1)=divide.test(i,4);
    a_extend_free_impact(i,2)=c1*(y(2,i)+c2)./(y(1,i)+c3);
    %a_extend_free(i,3)=abs((a_t(length(t))-divide.test(i,4))./divide.test(i,4));
    a_extend_free_impact(i,4)=t(i);
    a_extend_free_impact(i,5)=y(1,i);
    a_extend_free_impact(i,6)=y(2,i);
end
