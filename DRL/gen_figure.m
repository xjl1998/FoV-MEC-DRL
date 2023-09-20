function [] = gen_figure(title,x,y)

    clf;
    figure(1);

    plot(rd,power_records,'-r','LineWidth',2);
    hold on;
    plot(rd,power_records_noRis,'-b','LineWidth',2);
     %为背景添加网格
    grid on
    %添加图形名称
    title("优化后功率对比");
    %添加坐标轴名称
    xlabel("最低要求速率");
    ylabel("功率总和");
   
    %添加图例及文字说明
    legend('RIS-Assisted','No-RIS-Assisted ');


    figure(2);
        %为背景添加网格

     plot(rd,rate_records,'-r','LineWidth',2);
     hold on;
     plot(rd,rate_records_noRis,'-b','LineWidth',2);
 
    grid on
    %添加图形名称
    title("优化后速率对比");
    %添加坐标轴名称
    xlabel("最低要求速率");
    ylabel("实际速率总和");
    %添加图例及文字说明
    legend('RIS-Assisted','No-RIS-Assisted ');