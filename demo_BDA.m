aaa={'djc','jj','jl','ly','mhw','phl','sxy','wk','wsf','ww','wyw','xyl','ys','zjy'};
for j = 1:14
	Sub=aaa{j};
	srcStr = {[Sub,'LDSc1'],[Sub,'LDSc1'],[Sub,'LDSc2'],[Sub,'LDSc2'],[Sub,'LDSc3'],[Sub,'LDSc3']};
	tgtStr = {[Sub,'LDSc2'],[Sub,'LDSc3'],[Sub,'LDSc1'],[Sub,'LDSc3'],[Sub,'LDSc1'],[Sub,'LDSc2']};
	options.gamma = 1.0;
	options.lambda = 10.0;
	options.kernel_type = 'rbf';
	options.T = 10;
	options.dim = 100;
	options.mu = 0.5;
	options.mode = 'BDA';
	for iData = 1:6
     	src = char(srcStr{iData});
     	tgt = char(tgtStr{iData});
     	options.data = strcat(src,'_vs_',tgt);
     	% Preprocess data using Z-score
     	load(['/media/data/ld/BDA2018/preBDAdata/' src '.mat']);
     	fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
     	Xs = zscore(fts,1);    clear fts
     	Ys = labels;           clear labels
     	load(['/media/data/ld/BDA2018/preBDAdata/' tgt '.mat']);
     	fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
     	Xt = zscore(fts,1);    clear fts
     	Yt = labels;           clear labels
     	mean7=mean(Xs,1);
     	mean77=mean7(1);
     	[Acc,acc_ite,~] = MyBDA(Xs,Ys,Xt,Yt,options,src,tgt);
     	%fprintf('Acc:%.4f',Acc);
     	acc_ite
	end
end

