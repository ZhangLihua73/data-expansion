% clc
% clear
% %t(s) v(m/s) h(m) a(m/s^2) y(m) i maxlevel
% pathname='F:\ZLH\Basilisk\share\vertical\cell2\14bounce\';
% divide=load([pathname,'divide_bounce1.mat']);
err2=0.00251;
err3=0.029;
err4=3.47225E-8;
% err5=0.12728; 
c1=-0.00385;
c2=0.06531;
c3=0.08264;
c4=1.55184E-7;
c5=0;
c2_array=(c2-err2):2e-4:(c2+err2);
c3_array=(c3-err3):0.002:(c3+err3);
c4_array=(c4-err4):2e-9:(c4+err4);
% c2_array=(c2-err2):1e-4:(c2+err2);
% c3_array=(c3-err3):2e-9:(c3+err3);
% c5_array=(c5-err5):0.01:(c5+err5); 

tf=divide.test(length(divide.test(:,1)),1);%tf=0.03 6144000000000;

h=-0.000001;

%位移，速度
t0=divide.train(length(divide.train(:,1)),1);%
y0 = [divide.train(length(divide.train(:,1)),3);divide.train(length(divide.train(:,1)),2)];
af0 = divide.train(length(divide.train(:,1))-1,4);

h_end=divide.test(length(divide.test(:,1)),3);
v_end=divide.test(length(divide.test(:,1)),2);
a_end=divide.test(length(divide.test(:,1)),4);
double a_test;
t_array=t0:h:tf;
%%%%%%%%%%%%%%%有C4且不固定
% 起始与结束时刻不变，确定合适的c2、c3、c4误差值
err_a=zeros(length(c2_array),length(c3_array),length(c4_array));
for i=1:length(c2_array)
    for j=1:length(c3_array)
        for k=1:length(c4_array)
%             for n=1:length(c5_array)
                c2=c2_array(i);
                c3=c3_array(j);
                c4=c4_array(k);
%                 c5=c5_array(n);
                tspan = [t0, tf];
                [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5);
                for m=1:length(t)
                    a(m)=c1*(y(2,m)+c2)/(y(1,m)+c4)-28.00798232*(y(2,m)+c3)*(y(2,m)+c3)-8.63406;
                end
                err_a(i,j,k)=abs((a_end-a(length(t)))/a_end);
%             end
        end
    end
end 
err_min_a=1;
for i=1:length(c2_array)
    for j=1:length(c3_array)
        for k=1:length(c4_array)
%             for n=1:length(c5_array)
                if(err_a(i,j,k)<err_min_a)
                    err_min_a=err_a(i,j,k);
                    c2=c2_array(i);
                    c3=c3_array(j);
                    c4=c4_array(k);
%                     c5=c5_array(n);
                end
%             end
        end
    end
end
%结束时刻不变，改变起始时刻，确定外推误差最小的起始时刻
err_t=zeros(length(divide.train(:,1)),1);
for k=1:length(divide.train(:,1))
    t0=divide.train(k,1);
    tspan = [t0, tf];
    y0 = [divide.train(k,3);divide.train(k,2)];
    [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5);
    for m=1:length(t)
        a(m)=c1*(y(2,m)+c2)/(y(1,m)+c4)-28.00798232*(y(2,m)+c3)*(y(2,m)+c3)-8.63406;
    end
    err_t(k,1)=abs((a_end-a(length(t)))/a_end);
end
err_min_t=1;
for k=2:length(divide.train(:,1))
    if(err_t(k)<err_min_t)
        err_min_t=err_t(k);
        t0=divide.train(k,1);
        y0 = [divide.train(k,3);divide.train(k,2)];
%         af0 = divide.train(k-1,4);
    end
end
%%%%%%%%%扩展：与数值模拟时刻对比
h=-0.0000002;
tspan = [t0, tf];
[t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5);
% a_s,a_extend,error,t,h,v
a_extend=zeros(length(divide.test(:,1)),6);
%%%%%根据test对应时刻扩展
for i=1:length(divide.test(:,1))
    tf=divide.test(i,1);
    tspan = [t0, tf];
    [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5);
    for m=1:length(t)
        a_t(m)=c1*(y(2,m)+c2)/(y(1,m)+c4)-28.00798232*(y(2,m)+c3)*(y(2,m)+c3)-8.63406;
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

% %%%%%%%%%自由扩展，不用数值模拟数据进行测试
% h=-0.000001;
% tf=0.0361;
% tspan = [t0, tf];
% [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4);
% %a_s,a_extend,error,t,h,v
% a_extend_free_bounce1=zeros(length(t),6);
% %%%%%根据test对应时刻扩展
% for i=1:length(t)
%     %a_extend_free(i,1)=divide.test(i,4);
%     a_extend_free_bounce1(i,2)=c1*(y(2,i)+c2)./(y(1,i)+c3)+c4*(y(2,i)+0.209062)*(y(2,i)+0.209062)+0.063757245*y(3,i);
%     %a_extend_free(i,3)=abs((a_t(length(t))-divide.test(i,4))./divide.test(i,4));
%     a_extend_free_bounce1(i,4)=t(i);
%     a_extend_free_bounce1(i,5)=y(1,i);
%     a_extend_free_bounce1(i,6)=y(2,i);
% end
% save ([pathname,'extend_bounce.mat'],'a_extend','a_extend_free_bounce1');
