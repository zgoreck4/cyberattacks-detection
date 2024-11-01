startindex=max(find(t <= starttime));
stopindex=min(find(t >= stoptime));

time=t(startindex:stopindex);
ASinputx=ASinput(startindex:stopindex,:);
reac1x=reac1(startindex:stopindex,:);
reac2x=reac2(startindex:stopindex,:);
reac3x=reac3(startindex:stopindex,:);
reac4x=reac4(startindex:stopindex,:);
reac5x=reac5(startindex:stopindex,:);
settlerx=settler(startindex:stopindex,:);
inx=in(startindex:stopindex,:);

figure(3);
subplot(3,3,1);
plot(time,reac1x(:,3));
grid on;
title('XI, reactor 1');
subplot(3,3,2);
plot(time,reac2x(:,3));
grid on;
title('XI, reactor 2');
subplot(3,3,3);
plot(time,reac3x(:,3));
grid on;
title('XI, reactor 3');
subplot(3,3,4);
plot(time,reac4x(:,3));
grid on;
title('XI, reactor 4');
subplot(3,3,5);
plot(time,reac5x(:,3));
grid on;
title('XI, reactor 5');

subplot(3,3,6);
plot(time,(ASinputx(:,3)./ASinputx(:,15)));
grid on;
title('XI, input to AS');

subplot(3,3,7);
plot(time,settlerx(:,3));
grid on;
title('XI, underflow');
subplot(3,3,8);
plot(time,settlerx(:,19));
grid on;
title('XI, effluent');
subplot(3,3,9);
plot(time,inx(:,3));
grid on;
title('XI, influent');
