clc
clear
test_l=14;
pathname='F:\ZLH\Basilisk\share\super-test\case\result\3\bounce\';
mydata=load('F:\ZLH\Basilisk\share\super-test\case\result\3\bounce\bounce-level.dat');
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
pathname='F:\ZLH\Basilisk\share\super-test\case\result\3\bounce\';
mydata=load('F:\ZLH\Basilisk\share\super-test\case\result\3\bounce\bounce-level.dat');
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

% 初始化逻辑索引为所有true（保持所有行）
keep_rows = true(size(bounce1_data, 1), 1);

% 计算第11列中每一点与前后点的差值
for i = 2:(size(bounce1_data, 1) - 1)
    % 如果差值大于阈值，则将对应的逻辑索引设置为false（删除行）
    if (abs(bounce1_data(i, 4))) > (1+deviation_threshold)*(abs(bounce1_data(i+1, 4))) || (abs(bounce1_data(i, 4))) < (1-deviation_threshold)*(abs(bounce1_data(i-1, 4)))
        keep_rows(i) = false;
    end
end

% 使用逻辑索引过滤 `impact_data`
bounce1_data = bounce1_data(keep_rows, :);

save ([pathname,'data.mat'],'impact_data','bounce1_data')
result_a=load([pathname,'data.mat']);

%%%%%%%%%%%%%%%%%
% 指定文件保存路径路径和文件名前缀
filename_prefix = 'extend';

a_mul_max=5 ;
h_sample_max=4;
%c2,cerr2,c3,err3,,error,a_mul,h_sample,1+a_mul*0.5，length(samplingpoint)，length(train)，length(test)
temp=zeros(a_mul_max,h_sample_max,14);
para_ana=zeros(a_mul_max*h_sample_max,14);
for a_mul=1:a_mul_max
    for h_sample=1:h_sample_max
        %%%%%%%%%%回弹进行拟合、外推、采样
        %maxlevel,test-start,test-end
        bounce1_test_para=zeros(3,1);
        bounce1_test_para(2,1)=bounce1_data(1,6);
        bounce1_test_para(3,1)=bounce1_data(1,6);
        for i=2:length(bounce1_data(:,1))-1
            if(abs(bounce1_data(i,4))>=abs(bounce1_data(1,4)/(1+a_mul*0.5))&&abs(bounce1_data(i+1,4))<abs(bounce1_data(1,4)/(1+a_mul*0.5)))
                bounce1_test_para(1,1)=bounce1_data(1,7);
                bounce1_test_para(2,1)=bounce1_data(1,6);
                bounce1_test_para(3,1)=bounce1_data(i-1,6);
            end
        end
        test=zeros((bounce1_test_para(3,1)-bounce1_test_para(2,1)+1),7);
        train=zeros((length(bounce1_data(:,1))-length(test(:,1))),7);
        for j=1:7
            for i=1:length(test(:,1))
                test(i,j)=bounce1_data(i,j);
            end
        end
        train=zeros((length(bounce1_data(:,1))-length(test(:,1))),7);
        for j=1:7
            for i=1:length(train(:,1))
                train(i,j)=bounce1_data(i+length(test(:,1)),j);%t(s) v(m/s) h(m) a(m/s^2) y(m) i maxlevel
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
        %modelFunc = @(p,x) -0.0225.*(x(:,1)+p(1))./(x(:,2)+p(2));
        c1 = -0.025714286;
        c4 = 0.391654252;
        c6 = 0.75;
        modelFunc = @(p,x) +c1.*(x(:,1)+p(1))./(x(:,2)+p(2))-c4.*(x(:,1)-p(3)).*(x(:,1)-p(3))-c6;
        % 创建一组初始参数估计值
        initialParams = [0, 0, 0]; % 必须替换为合理的初始猜测值

        % 拟合模型
        mdl = fitnlm([x1 x2], y, modelFunc, initialParams);

        % 提取参数估计值和标准差
        params = mdl.Coefficients{:, 'Estimate'};      % 参数估计值
        stdErrors = mdl.Coefficients{:, 'SE'};         % 参数标准差

        % 分别取出c2和c3及它们的标准差
        
        c2 = params(1);
        c3 = params(2);
        c5 = params(3);
        c2_std = stdErrors(1);
        c3_std = stdErrors(2);
        c5_std = stdErrors(3);



        fprintf('c2: %f (Standard deviation: %f)\n', c2, c2_std);
        fprintf('c3: %f (Standard deviation: %f)\n', c3, c3_std);
        fprintf('c5: %f (Standard deviation: %f)\n', c5, c5_std);
        
        
        % 生成一个等差序列数组
        linspace_array = linspace(-1, 1, 20);
        
        % 对数组进行线性插值操作
        c2_array = c2 + c2_std * linspace_array;
        c3_array = c3 + c3_std * linspace_array;
        c5_array = c5 + c5_std * linspace_array;
        temp(a_mul,h_sample,1)=c2;
        temp(a_mul,h_sample,2)=c2_std;
        temp(a_mul,h_sample,3)=c3;
        temp(a_mul,h_sample,4)=c3_std;
        temp(a_mul,h_sample,13)=c5;
        temp(a_mul,h_sample,14)=c5_std;
        temp(a_mul,h_sample,6)=a_mul;
        temp(a_mul,h_sample,7)=h_sample;
        %%%%%%%%%%%%%%最佳参数确定
        tf=test(1,1);%tf=0.036144000000000;
        
        h=-0.00001;
        
        %位移，速度
        t0=train(1,1);%t0=0.035392600000000;
        y0 = [train(1,3);train(1,2)];
        %位移，速度
        t0=train(length(train(:,1)),1);%
        y0 = [train(length(train(:,1)),3);train(length(train(:,1)),2)];
        af0 = train(length(train(:,1))-1,4);
        h_end=test(1,3);
        v_end=test(1,2);
        a_end=test(1,4);
        t_array=t0:h:tf;
        h_end=test(length(test(:,1)),3);
        v_end=test(length(test(:,1)),2);
        a_end=test(length(test(:,1)),4);
        t_array=t0:h:tf;
        %起始与结束时刻不变，确定合适的c1、c2误差值
        err_a=zeros(length(c2_array),length(c3_array),length(c5_array));
        for i=1:length(c2_array)
            for j=1:length(c3_array)
                for k=1:length(c5_array)
                    c2=c2_array(i);
                    c3=c3_array(j);
                    c5=c5_array(k);
                    tspan = [t0, tf];
                    [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
                    for m=1:length(t)
                        a(m)=c1*(y(2,m)+c2)/(y(1,m)+c3)-c4*(y(2,m)+c5)*(y(2,m)+c5)-c6;
                    end
                    err_a(i,j,k)=abs((a_end-a(length(t)))/a_end);
                end
            end
        end
        err_min_a=1;
        for i=1:length(c2_array)
            for j=1:length(c3_array)
                for k=1:length(c5_array)
                    if(err_a(i,j,k)<err_min_a)
                        err_min_a=err_a(i,j,k);
                        c2=c2_array(i);
                        c3=c3_array(j);
                        c5=c5_array(k);
                    end
                end
            end
        end
        %结束时刻不变，改变起始时刻，确定外推误差最小的起始时刻
        err_t=zeros(length(train(:,1)),1);
        for k=1:length(train(:,1))
            t0=train(k,1);
            tspan = [t0, tf];
            y0 = [train(k,3);train(k,2)];
            [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
            for m=1:length(t)
                a(m)=c1*(y(2,m)+c2)/(y(1,m)+c3)-c4*(y(2,m)+c5)*(y(2,m)+c5)-c6;
            end
            err_t(k,1)=abs((a_end-a(length(t)))/a_end);
        end
        err_min_t=1;
        for k=2:length(train(:,1))
            if(err_t(k)<err_min_t)
                err_min_t=err_t(k);
                t0=train(k,1);
                y0 = [train(k,3);train(k,2)];
            end
        end
        %%%%%%%%%扩展：与数值模拟时刻对比
        h=-0.00001;
        tspan = [t0, tf];
        [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
        % a_s,a_extend,error,t,h,v
        a_extend=zeros(length(test(:,1)),6);
        %%%%%根据test对应时刻扩展
        for i=1:length(test(:,1))
            tf=test(i,1);
            tspan = [t0, tf];
            [t,y] = RK4(@f, tspan, y0, h, c1, c2, c3, c4, c5, c6);
            for m=1:length(t)
                a_t(m)=c1*(y(2,m)+c2)/(y(1,m)+c3)-c4*(y(2,m)+c5)*(y(2,m)+c5)-c6;
            end
            a_extend(i,1)=test(i,4);
            a_extend(i,2)=a_t(length(t));
            a_extend(i,3)=abs((a_t(length(t))-test(i,4))./test(i,4));
            a_extend(i,4)=tf;
            a_extend(i,5)=y(1,length(t));
            a_extend(i,6)=y(2,length(t));
        end
        %c2,cerr2,c3,err3,,error,a_mul,h_sample,1+a_mul*0.5，length(samplingpoint)，length(train)，length(test)
        temp(a_mul,h_sample,5)=max(a_extend(:,3));
        temp(a_mul,h_sample,8)=length(a_extend(:,3));
        temp(a_mul,h_sample,9)=1+a_mul*0.5;
        para_ana((a_mul-1)*3+h_sample, :) = temp(a_mul,h_sample,:);
        % 构建完整的文件名
        filename = [filename_prefix, '-', num2str(a_mul),'-', num2str(h_sample), '.mat'];
        % 这里是你需要保存的变量 a_extend
        % 保存变量为 mat 文件
        save(fullfile(pathname, filename), 'a_extend', 'samplingpoint', 'train', 'test');
    end
end
save ([pathname,'extend_bounce.mat'],'para_ana','temp');
