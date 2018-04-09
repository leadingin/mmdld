aaa={'djc','jj','jl','ly','mhw','phl','sxy','wk','wsf','ww','wyw','xyl','ys','zjy'};
for j = 1:14
	Sub=aaa{j};
	%Sub='djc'
	srcStr = [Sub,'s'];
	tgtStr = [Sub,'t'];
	options.gamma = 1.0;
	options.lambda = 10.0;
	options.kernel_type = 'rbf';
	options.T = 10;
	options.dim = 100;
	options.mu = 0.3;
	options.mode = 'W-BDA';
	src = char(srcStr);
	tgt = char(tgtStr);
	options.data = strcat(src,'_vs_',tgt);
	% Preprocess data using Z-score
	load(['/media/data/ld/BDA2018/preBDAdata2to1/' src '.mat']);
	fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
	Xs = zscore(fts,1);    clear fts
	Ys = labels;           clear labels
	load(['/media/data/ld/BDA2018/preBDAdata2to1/' tgt '.mat']);
	fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
	Xt = zscore(fts,1);    clear fts
	Yt = labels;           clear labels
	[Acc,acc_ite,~] = MyBDA1(Xs,Ys,Xt,Yt,options,src,tgt);
	%fprintf('Acc:%.4f',Acc);
	acc_ite
end
