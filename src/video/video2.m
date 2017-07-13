

Meanx = zeros(184,46);
Meany = zeros(184,46);
distance = zeros(184,46);
Lnorm = zeros(184,41);
angle = zeros(184,46);
d = dir('*.txt');  fmt = repmat('%f',1,140);
for ii = 1: 184
 fid = fopen([ num2str(ii) '_CLNF_features' '.txt'], 'rt');
 datacell  = textscan(fid, fmt, 'Delimiter', ',', 'HeaderLines', 1, 'CollectOutput', 1);
 data = datacell{1};
 x = data(:,5:72);list_o_cols_to_delete = [1:18 29:32];  % 46 stable points 
 x(:,list_o_cols_to_delete) = [];
 y = data(:,73:140);y(:,list_o_cols_to_delete) = [];  % 46 stable points 

 
  
  for i=1:size(x,2)
      D(:,i)= sqrt(sum((x(:,i)-y(:,i)) .^ 2));
  end
   
 
 for i=1:size(x,2)
     c = x(:,i);
     d = y(:,i);
     theta(:,i) = acos(min(1,max(-1, c(:).' * d(:) / norm(c) / norm(d) )));
 end

  angle(ii,:) = theta; % Angle between coordinates (46 features)
  distance(ii,:)= D; % Euclidean distance between coordinates (46 features)
  [row, col] = find(isnan(distance));
  
  Meanx(ii,:)= mean(x);
  Meany(ii,:)= mean(y);
  dif = Meanx-Meany; % (46 features)
  dif = dif(all(~isnan(dif),2),:);
  list = [1:5 16:21]; Left = dif(:,list );
  list = [6:10 22:27]; Right = dif(:,list );
  list = [28:46]; Mouth = dif(:,list );
  Lnorm1 = sqrt(sum(abs(Left).^2));
  Lnorm2 = sqrt(sum(abs(Right).^2));
  Lnorm3 = sqrt(sum(abs(Mouth).^2));
  Lnorm(ii,:) = [Lnorm1,Lnorm2,Lnorm3]; %(41 features)
 

end
row = [65 93 127];
distance (row,:) = [];
Lnorm(row,:)=[];
angle(row,:) = [];
 
video = [distance, angle, Lnorm];
normA = max(video) - min(video);               % this is a vector
normA = repmat(normA, [length(video) 1]);  % this makes it a matri                                    
normalizedA = video./normA; 
filename = 'Normalizedvideo.xlsx';
Lnorm1 = Lnorm(:,1:11);
Lnorm2 =Lnorm(:,12:22);
Lnorm3 = Lnorm(:,23:end);

filename2 = 'video.xlsx';
filename3='distance.xlsx';filename4='angle.xlsx';filename5='Lnorm.xlsx';

xlswrite(filename,normalizedA);
xlswrite(filename2,video);
xlswrite(filename3,distance);xlswrite(filename4,angle);xlswrite(filename5,Lnorm);
  
%%
normA = max(fea_video) - min(fea_video);               % this is a vector
normA = repmat(normA, [length(fea_video) 1]);  % this makes it a matri                                    
normalizedA = fea_video./normA;
fea_video = normalizedA;