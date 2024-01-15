clc
clear
pathname='F:\ZLH\Basilisk\share\super-test\case\result\3\rand\';
mydata=load('F:\ZLH\Basilisk\share\super-test\case\result\3\rand\bounce-level.dat');
data_para=zeros(5,1);%impact-end(length(impact)),bounce-start,bounce-peak,length(impact),length(bounce)
bounce_start1=1347;%%%%%%%%%%%%%
bounce_start2=2355;%%%%%%%%%%%%%
data_para(1,1)=bounce_start1-1;
data_para(2,1)=bounce_start1;
data_para(4,1)=data_para(1,1);
for i=bounce_start1:bounce_start2-1
    if(mydata(i,8)<0.&&mydata(i-1,8)>0.)
        data_para(3,1)=i-1;
    end
end
data_para(5,1)=data_para(3,1)-data_para(2,1)+1;
impact=zeros(data_para(4,1),length(mydata(1,:)));
bounce1=zeros(data_para(5,1),length(mydata(1,:)));
%%%%%%%数据填充
for j=1:length(mydata(1,:))
    for i=1:length(impact(:,1))
        impact(i,j)=mydata(i,j);
    end
    for i=1:length(bounce1(:,1))
        bounce1(i,j)=mydata(data_para(1,1)+i,j);
    end
end
%%%%%%%
mydata=load('F:\ZLH\Basilisk\share\super-test\case\result\3\rand\bounce-level.dat');
d=1;
impact_start=0;
impact_end=data_para(1,1);
bounce1_start=0;
bounce1_end=0;
parameter=zeros(6,1);
%根据加速度正负交替选取润滑力作用范围
%impact加速度选取
for i=2:length(impact(:,1))%润滑力不会从一开始主导
    if(impact(i,11)>0.&&impact(i,8)<0.)
        impact_end=i;
        if(i>impact_end)
            impact_end=i;
        end
    end
    if(impact(i,11)>0.&&impact(i-1,11)<0.&&impact(i,8)<0.)
        impact_start=i;
    end
    impact_length=impact_end-impact_start+1;
end
%回弹-根据球回弹到最高点选取数据
inf=2.*impact(length(impact(:,11)),11);
gra=-0.75;
bounce1_start=1;
bounce1_end=length(bounce1(:,1));
for i=2:length(bounce1(:,1))-1
    if(abs(bounce1(i,11))>inf)
        if(bounce1_start<i)
            bounce1_start=i+1;
        end
    end   
end
bounce1_start=bounce1_start+1;;
bounce1_length=bounce1_end-bounce1_start+1;

parameter(1,1)=impact_start;
parameter(2,1)=impact_end;
parameter(3,1)=impact_length;
parameter(4,1)=bounce1_start;
parameter(5,1)=bounce1_end;
parameter(6,1)=bounce1_length;
impact_data=zeros(impact_length,7);
bounce1_data=zeros(bounce1_length,7);
%%%%%%%%%%%%
%下落-加速度正负交替选取
for i=1:length(impact_data(:,1))
    impact_data(i,1)=impact(i+impact_start-1,2);%t(s)
    impact_data(i,2)=impact(i+impact_start-1,8);%v(m/s)
    impact_data(i,3)=impact(i+impact_start-1,5)-d/2.;%h(m)
    impact_data(i,4)=impact(i+impact_start-1,11);%a(m/s^2)
    impact_data(i,5)=impact(i+impact_start-1,5);%y(m)
    impact_data(i,6)=impact(i+impact_start-1,1);%i
    impact_data(i,7)=impact(i+impact_start-1,16);%maxlevel
end
% 阈值 - 你需要根据数据设置这个阈值
deviation_threshold = 0.2;

% 初始化逻辑索引为所有true（保持所有行）
keep_rows = true(size(impact_data, 1), 1);

% 计算第11列中每一点与前后点的差值
for i = 2:(size(impact_data, 1) - 1)
    % 如果差值大于阈值，则将对应的逻辑索引设置为false（删除行）
    if (abs(impact_data(i, 4))) > (1+deviation_threshold)*(abs(impact_data(i+1, 4))) || (abs(impact_data(i, 4))) < (1-deviation_threshold)*(abs(impact_data(i-1, 4)))
        keep_rows(i) = false;
    end
end

% 使用逻辑索引过滤 `impact_data`
impact_data = impact_data(keep_rows, :);

% 你可以在这里进行保存或其他处理

%回弹-距离壁面一个直径选取
for i=1:length(bounce1_data(:,1))
    bounce1_data(i,1)=bounce1(i+bounce1_start-1,2);%t(s)
    bounce1_data(i,2)=bounce1(i+bounce1_start-1,8);%v(m/s)
    bounce1_data(i,3)=bounce1(i+bounce1_start-1,5)-d/2.;%h(m)
    bounce1_data(i,4)=bounce1(i+bounce1_start-1,11);%a(m/s^2)
    bounce1_data(i,5)=bounce1(i+bounce1_start-1,5);%y(m)
    bounce1_data(i,6)=bounce1(i+bounce1_start-1,1);%i
    bounce1_data(i,7)=bounce1(i+bounce1_start-1,16);%maxlevel
end
save ([pathname,'data.mat'],'impact_data','bounce1_data')
%%%%%%%%%%%%%%%%a加随机误差
load([pathname,'data.mat']);
% 设置随机误差的标准差
noiseStd = 0.5;
% 生成与其长度一样多的随机误差
noise = noiseStd * randn(size(impact_data, 1), 1);
% 将随机误差添加到第4列——加速度
impact_data(:, 4) = impact_data(:, 4) + noise;
% 生成与其长度一样多的随机误差
noise = noiseStd * randn(size(bounce1_data, 1), 1);
bounce1_data(:, 4) = bounce1_data(:, 4) + noise;
% 检查结果
save ([pathname,'data_rand_a.mat'],'impact_data','bounce1_data')
%%%%%%%%%%%%%%%%%%%%%%t加随机误差
load([pathname,'data.mat']);
% 假设 impact_data 是已经存在的变量，这里不做初始化
% 设置随机误差的标准差
noiseStd = 1e-4;
% 生成与其长度一样多的随机误差
noise = noiseStd * randn(size(impact_data, 1), 1);
% 将随机误差添加到第4列——时间
impact_data(:, 1) = impact_data(:, 1) + noise;
% 生成与其长度一样多的随机误差
noise = noiseStd * randn(size(bounce1_data, 1), 1);
bounce1_data(:, 1) = bounce1_data(:, 1) + noise;
save ([pathname,'data_rand_t.mat'],'impact_data','bounce1_data')
clc
clear
pathname='F:\ZLH\Basilisk\share\super-test\case\result\3\rand\';
original=load([pathname,'data.mat']);
a=load([pathname,'data_rand_a.mat']);
t=load([pathname,'data_rand_t.mat']);
figure;
plot(original.impact_data(:, 1), original.impact_data(:, 4),'k-', 'LineWidth', 1.5);
hold on;
plot(t.impact_data(:, 1), t.impact_data(:, 4),'b-', 'LineWidth', 1.5);
hold on;
plot(a.impact_data(:, 1), a.impact_data(:, 4),'r-','LineWidth', 1.5);
hold off;
xlabel('Time(s)', 'FontSize', 24, 'FontName', 'Times New Roman');
ylabel('Acceleration(m/s^2)', 'FontSize', 24, 'FontName', 'Times New Roman');
title('Add random error to falling data', 'FontSize', 24, 'FontName', 'Times New Roman');
legend('Numerical simulation data','Add random error to time','Add random error to acceleration', 'FontSize', 24, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 24, 'FontName', 'Times New Roman');
figure;
plot(original.bounce1_data(:, 1), original.bounce1_data(:, 4),'k-', 'LineWidth', 1.5);
hold on;
plot(t.bounce1_data(:, 1), t.bounce1_data(:, 4),'b-', 'LineWidth', 1.5);
hold on;
plot(a.bounce1_data(:, 1), a.bounce1_data(:, 4),'r-','LineWidth', 1.5);
hold off;
xlabel('Time(s)', 'FontSize', 24, 'FontName', 'Times New Roman');
ylabel('Acceleration(m/s^2)', 'FontSize', 24, 'FontName', 'Times New Roman');
title('Add random error to bouncing data', 'FontSize', 24, 'FontName', 'Times New Roman');
legend('Numerical simulation data','Add random error to time','Add random error to acceleration', 'FontSize', 24, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 24, 'FontName', 'Times New Roman');

%%%%%%%%%%%%%%%%%
% 指定文件保存路径路径和文件名前缀
filename_prefix = 'extend';

a_mul_max=50;%1.1\1.2\1.3\1.4……5
h_sample_max=4;
%c2,cerr2,c3,err3,,error,a_mul,h_sample,1+a_mul*0.5，length(samplingpoint)，length(train)，length(test)
temp=zeros(a_mul_max,h_sample_max,12);
para_ana=zeros(a_mul_max*h_sample_max,12);
for a_mul=1:a_mul_max
    for h_sample=1:h_sample_max
        %t(s) v(m/s) h(m) a(m/s^2) y(m) i maxlevel
        pathname='F:\ZLH\Basilisk\share\super-test\case\result\3\rand\';
        divide=load([pathname,'divide_impact.mat']);
        %%%%%%%%%%下落进行拟合、外推、采样
        %maxlevel,test-start,test-end
        test_para=zeros(3,1);
        for i=2:length(impact_data(:,1))
            if(abs(impact_data(i,4))>=abs(impact_data(length(impact_data(:,1)),4)/(1+a_mul*0.1))&&abs(impact_data(i-1,4))<abs(impact_data(length(impact_data(:,1)),4)/(1+a_mul*0.1)))
                test_para(1,1)=impact_data(i,7);
                test_para(2,1)=impact_data(i,6);
                test_para(3,1)=impact_data(length(impact_data(:,1)),6);
            end
        end
        test=zeros((test_para(3,1)-test_para(2,1)+1),7);
        train=zeros((length(impact_data(:,1))-length(test(:,1))),7);
        for j=1:7
            for i=1:length(test(:,1))
                test(i,j)=impact_data(i+length(train(:,1)),j);
            end
        end
        train=zeros((length(impact_data(:,1))-length(test(:,1))),7);
        for j=1:7
            for i=1:length(train(:,1))
                train(i,j)=impact_data(i,j);%t(s) v(m/s) h(m) a(m/s^2) y(m) i maxlevel
            end
        end
        
        samplingpoint=zeros(int32(length(train(:, 1))/h_sample),7);
        for i=1:7
            for j=1:length(samplingpoint(:,1))
                samplingpoint(j,i)=train(1+(j-1)*h_sample, i);
            end
        end
        
        %%%%%%%%%拟合
        % 假定 x1, x2 和 y 是你的数据
        x1 = samplingpoint(:,2); % x1 数据向量
        x2 = samplingpoint(:,3); % x2 数据向量
        y = samplingpoint(:,4);  % y 数据向量
        % 拟合方程模型。创建一个匿名函数，p(1)对应于c2，p(2)对应于c3
        modelFunc = @(p,x) -0.0225.*(x(:,1)+p(1))./(x(:,2)+p(2));

        % 创建一组初始参数估计值
        initialParams = [1, 1]; % 必须替换为合理的初始猜测值

        % 拟合模型
        mdl = fitnlm([x1 x2], y, modelFunc, initialParams);

        % 提取参数估计值和标准差
        params = mdl.Coefficients{:, 'Estimate'};      % 参数估计值
        stdErrors = mdl.Coefficients{:, 'SE'};         % 参数标准差

        % 分别取出c2和c3及它们的标准差
        c1 = -0.0225;
        c2 = params(1);
        c3 = params(2);
        c2_std = stdErrors(1);
        c3_std = stdErrors(2);
        c4=0;
        c5=0;
        c6=0;


        fprintf('c2: %f (Standard deviation: %f)\n', c2, c2_std);
        fprintf('c3: %f (Standard deviation: %f)\n', c3, c3_std);
        
        
        % 生成一个等差序列数组
        linspace_array = linspace(-1, 1, 50);
        
        % 对数组进行线性插值操作
        c2_array = c2 + c2_std * linspace_array;
        c3_array = c3 + c3_std * linspace_array;
        temp(a_mul,h_sample,1)=c2;
        temp(a_mul,h_sample,2)=c2_std;
        temp(a_mul,h_sample,3)=c3;
        temp(a_mul,h_sample,4)=c3_std;
        temp(a_mul,h_sample,6)=a_mul;
        temp(a_mul,h_sample,7)=h_sample;
        %%%%%%%%%%%%%%最佳参数确定
        tf=test(1,1);%tf=0.036144000000000;
        
        h=0.00001;
        
        %位移，速度
        t0=train(1,1);%t0=0.035392600000000;
        y0 = [train(1,3);train(1,2)];
        
        h_end=test(1,3);
        v_end=test(1,2);
        a_end=test(1,4);
        t_array=t0:h:tf;
        %起始与结束时刻不变，确定合适的c1、c2误差值
        err_a=zeros(length(c2_array),length(c3_array));
        for i=1:length(c2_array)
            for j=1:length(c3_array)
                c2=c2_array(i);
                c3=c3_array(j);
                tspan = [t0, tf];
                [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
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
        %%%%%%%%%%%%%%%%%确定外推最佳起始时刻
        err_t=zeros(length(train(:,1)));
        for k=1:length(train(:,1))
            t0=train(k,1);
            tspan = [t0, tf];
            y0 = [train(k,3);train(k,2)];
            [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
            for m=1:length(t)
                a(m)=c1*(y(2,m)+c2)./(y(1,m)+c3);
            end
            err_t(k)=abs((a_end-a(length(t)))/a_end);
        end
        err_min_t=1;
        for k=1:length(train(:,1))
            if(err_t(k)<err_min_t)
                err_min_t=err_t(k);
                t0=train(k,1);
                y0 = [train(k,3);train(k,2)];
            end
        end
        tspan = [t0, tf];
        [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
        %a_s,a_extend,error,t,h,v
        a_extend=zeros(length(test(:,1)),6);
        
        for i=1:length(test(:,1))
            tf=test(i,1);
            tspan = [t0, tf];
            [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
            for k=1:length(t)
                a_t(k)=c1*(y(2,k)+c2)./(y(1,k)+c3);
            end
            a_extend(i,1)=test(i,4);
            a_extend(i,2)=a_t(length(t));
            a_extend(i,3)=abs((a_t(length(t))-test(i,4))./test(i,4));
            a_extend(i,4)=tf;
            a_extend(i,5)=y(2,length(t));
            a_extend(i,6)=y(1,length(t));
        end
        
        %c2,cerr2,c3,err3,,error,a_mul,h_sample,1+a_mul*0.5，length(samplingpoint)，length(train)，length(test)
        temp(a_mul,h_sample,5)=max(a_extend(:,3));
        temp(a_mul,h_sample,8)=length(a_extend(:,3));
        temp(a_mul,h_sample,9)=1+a_mul*0.5;
        temp(a_mul,h_sample,10)=length(samplingpoint(:,1));
        temp(a_mul,h_sample,11)=length(train(:,1));
        temp(a_mul,h_sample,12)=length(test(:,1));
        para_ana((a_mul-1)*3+h_sample, :) = temp(a_mul,h_sample,:);
        % 构建完整的文件名
        filename = [filename_prefix, '-', num2str(a_mul),'-', num2str(h_sample), '.mat'];
        % 保存变量为 mat 文件
        save(fullfile(pathname, filename), 'a_extend', 'samplingpoint', 'train', 'test');
    end
end
save ([pathname,'extend_impact.mat'],'para_ana','temp');
