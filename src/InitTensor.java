package LFT.cn.edu.swu;

import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.StringTokenizer;

public class InitTensor {
	public HashMap<Integer,HashMap<Integer,HashMap<Integer,ArrayList<Double>>>> ad=new HashMap<>();
	//initialize tensor:data U S T Udown Sdown Tdown  R
	public int rank=20;
	public int trainRound=6000;
	
	public ArrayList<TensorTuple> trainData=new ArrayList<>();
	public ArrayList<TensorTuple> testData=new ArrayList<>();
	
	public int trainDataCount=0;
	public int testDataCount=0;
	
	public int maxuid=0,maxsid=0,maxtid=0;
	public int minuid=Integer.MAX_VALUE,minsid=Integer.MAX_VALUE,mintid=Integer.MAX_VALUE;
	public double maxValue=0,minValue=Double.MAX_VALUE;
	
	public String trainFilePath=null;
	public String trainFilePath1=null;
	public String LaplaceFilePath=null;
	public String testFilePath=null;
	public String seperator=null;
	
	public double[][] U,S,T,U2,S2,T2,U3,S3,T3;
	public double[] a,b,c,w,m;
	public int [] count;
	public double [] square_,Mae_count;

//	None_Liner
	public double[][] a_1,b_1,c_1;

//	PSO
    public  double[][] pbest,weight,weight_beta;
	public  double[] gbest;
	public  int swarmNum = 100;
	public  double[] FitnessRMSEpbest;
	public  double FitnessRMSEgbest,precious_errors=1e10;
	public double c1,r1,c2,r2,omega;
	public  double[] tempRMSE;
	public double[][] Uup,Udown,Sup,Sdown,Tup,Tdown;
	public double[] aup,adown,bup,bdown,cup,cdown;
	
	//test error
	public double[] everyRoundMAE,everyRoundRMSE,RoundMae_1,RoundMae_2,RoundRmse_1,RoundRmse_2,minRMSE_,minMAE_;
	public double minRMSE=Double.MAX_VALUE,minMAE=Double.MAX_VALUE,minRMSE_1=1e10,minMAE_1=1e10,minRMSE_2=1e10,minMAE_2=1e10;
	public int minRMSERound=0,minMAERound=0,minRMSERound_1=0,minMAERound_1=0,minRMSERound_2=0,minMAERound_2=0;
	public int[][] D;
	public double[][] L;
	public double [][] LLT;
	public InitTensor(String trainFilePath,String testFilePath,String seperator) {
		this.trainFilePath=trainFilePath;
		this.testFilePath=testFilePath;
		this.seperator=seperator;
	}
	
	//init data
	public void initData(String FilePath,ArrayList<TensorTuple> list,int T) throws IOException {
		//creat file reader
		File input=new File(FilePath);
		BufferedReader in=new BufferedReader(new FileReader(input));
		String s=null;
		int count_1=0,count_2=0;
		while((s=in.readLine())!=null) {
			StringTokenizer st=new StringTokenizer(s,seperator);
			//StringTokenizer st=new StringTokenizer(s);
			
			String i=null;
			if(st.hasMoreTokens())
				i=st.nextToken();
			String j=null;
			if(st.hasMoreTokens())
				j=st.nextToken();
			String k=null;
			if(st.hasMoreTokens())
				k=st.nextToken();
			String v=null;
			if(st.hasMoreTokens())
				v=st.nextToken();
			//转换存储
			int uid=Integer.parseInt(i.replaceAll("\\.0", ""));
			int sid=Integer.parseInt(j.replaceAll("\\.0", ""));
			int tid=Integer.parseInt(k);
			if (v.equals("nan"))
			{
				continue;
			}

			double value=Double.parseDouble(v);
			
			TensorTuple t=new TensorTuple();
			t.uid=uid-1;
			t.sid=sid-1;
			t.tid=tid;
//			if (tid==1){
//				value=(value-5)/(1000-5);//归一化
//				System.out.println(value);
//				count_1++;
//				value=value*(1000-5)+5;
//			}
			if (tid==1)
			{
//				value=(value-3)/(1000-3);
//				count_2++;
				value=value*1;
			}
			t.value=value;
			list.add(t);
//			信息
			maxuid=maxuid>uid?maxuid:uid;
			maxsid=maxsid>sid?maxsid:sid;
			maxtid=maxtid>tid?maxtid:tid;
			maxValue=maxValue>value?maxValue:value;

			minuid=minuid<uid?minuid:uid;
			minsid=minsid<sid?minsid:sid;
			mintid=mintid<tid?mintid:tid;
			minValue=minValue<value?minValue:value;
//			计数
			if(T==1)
				trainDataCount++;
			if(T==2)
				testDataCount++;

		}
//		System.out.println(maxuid);
//		System.out.println(count_1);
	}
	public void initLFAData(String FilePath,ArrayList<TensorTuple> list,int T) throws IOException {
		//creat file reader
		File input=new File(FilePath);
		BufferedReader in=new BufferedReader(new FileReader(input));
		String s=null;
		int count_1=0,count_2=0;
		while((s=in.readLine())!=null) {
			StringTokenizer st=new StringTokenizer(s,seperator);
			//StringTokenizer st=new StringTokenizer(s);

			String i=null;
			if(st.hasMoreTokens())
				i=st.nextToken();
			String j=null;
			if(st.hasMoreTokens())
				j=st.nextToken();
			String v=null;
			if(st.hasMoreTokens())
				v=st.nextToken();
			//转换存储
			int uid=Integer.parseInt(i.replaceAll("\\.0", ""));
			int sid=Integer.parseInt(j.replaceAll("\\.0", ""));
			double value=Double.parseDouble(v);

			TensorTuple t=new TensorTuple();
			t.uid=uid-1;
			t.sid=sid-1;
//			if (tid==1){
//				value=(value-5)/(1000-5);//归一化
//				System.out.println(value);
//				count_1++;
//				value=value*(1000-5)+5;
//			}
//			if (tid==2)
//			{
//				value=(value-3)/(1000-3);
//				count_2++;
//				value=value*(1000-3)+3;
//			}
			t.value=value;
			list.add(t);
//			信息
			maxuid=maxuid>uid?maxuid:uid;
			maxsid=maxsid>sid?maxsid:sid;
			maxValue=maxValue>value?maxValue:value;

			minuid=minuid<uid?minuid:uid;
			minsid=minsid<sid?minsid:sid;
			minValue=minValue<value?minValue:value;
//			计数
			if(T==1)
				trainDataCount++;
			if(T==2)
				testDataCount++;

		}
//		System.out.println(maxuid);
//		System.out.println(count_1);
	}
	
	//初始化隐特征矩阵
	public double initScale=0.004;
	public int scale=1000;

	public void initD(){
		D=new int[maxsid][maxsid];
		for (int i = 0; i < maxsid-1; i++) {
			D[i][i]=-1;
			D[i+1][i]=1;
		}
	}
	public void initLaplace(String fileName) throws IOException {
		L = new double[maxuid][maxuid];
		LLT=new double[maxuid][maxuid];
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));
		String tempVoting = "";
		String str = "";
		int Num_trainData = 0;
		while ((tempVoting = in.readLine()) != null) {

			str = "" + tempVoting;
			String[] s = str.split("::");
			for (int j = 0; j < maxuid; j++) {
				L[Num_trainData][j] = Double.valueOf(s[j]);
			}
//			System.out.println(Num_trainData);
			Num_trainData++;
		}
		double [][]Lt =new double[maxuid][maxuid];
		for (int i1 = 0; i1 < maxuid; i1++) {
			for (int j = 0; j < maxuid; j++) {
				Lt[j][i1] = L[i1][j];
			}
		}
		for (int i1 = 0; i1 < maxuid; i1++) {
			for (int j = 0; j < maxuid; j++) {
				LLT[i1][j] = 0;
				for (int l = 0; l < maxuid; l++) {
					LLT[i1][j] += L[i1][l] * Lt[l][j];
				}
			}
		}
	}
	public double Fitness(double []weight,int index){
		double rmse=0,mae=0;
		int count=0;
		for (TensorTuple e : testData) {
			if(index==e.tid){
				e.valueHat = getPre(e.uid, e.sid, e.tid);
				count++;
				double v1=e.value,v2=e.valueHat;
				rmse+=Math.pow(v1-v2,2);
				mae+=Math.abs(v1-v2);
			}
		}


		return 0;
	}
	public  void initialPSO()
	{
		pbest = new double[swarmNum][maxtid];
		gbest = new double[maxtid];
		weight=new double[swarmNum][maxtid];
		weight_beta=new double[swarmNum][maxtid];
		Random random = new Random(System.currentTimeMillis());
		for (int j = 0; j < swarmNum; j++)
		{
			for (int i = 0; i < maxtid; i++) {
				weight[j][i]=0.1+random.nextDouble()*(maxtid-0.1);
				weight_beta[j][i]=random.nextDouble()*((maxtid-0.1) * 0.2);
				pbest[j][i]=random.nextDouble()*(maxtid);
				gbest[i]=1;
			}
		}
	}
	public void initialfitness()
	{
		FitnessRMSEgbest = 1e10;
		tempRMSE = new double[swarmNum];
		FitnessRMSEpbest = new double[swarmNum];
		for (int j = 0; j < swarmNum; j++)
		{
			FitnessRMSEpbest[j] = 1e10;
			tempRMSE[j] = 0;
		}
	}
	public void updatebestRMSE(double [] tempPbestRMSE) {
		for (int k = 0; k < swarmNum; k++) {
			if (tempPbestRMSE[k] < FitnessRMSEpbest[k]) {
				FitnessRMSEpbest[k] = tempPbestRMSE[k];
				for (int i = 0; i < maxtid; i++) {
					pbest[k][i] = weight[k][i];
				}
			}
			if (tempPbestRMSE[k] < FitnessRMSEgbest) {
				FitnessRMSEgbest = FitnessRMSEpbest[k];
				for (int i = 0; i < maxtid; i++) {
					gbest[i] = weight[k][i];
				}
			}
		}
	}
	public void containMinUST(double[][]U,double[][]S,double[][]T){
		for(int i=0;i<maxuid+1;i++) {
			for(int r=0;r<rank+1;r++) {
				U2[i][r]=U[i][r];
			}

		}
		for(int i=0;i<maxsid+1;i++) {
			for(int r=0;r<rank+1;r++) {
				S2[i][r]=S[i][r];
			}
		}
		for(int i=0;i<maxtid+1;i++) {
			for(int r=0;r<rank+1;r++) {
				T2[i][r]=T[i][r];
			}
		}

	}

	public void updataWeightAndWeight_beta(){
		for (int swarmNum_index=0;swarmNum_index<swarmNum;swarmNum_index++) {
			double sum=0;
			for (int i = 0; i < maxtid; i++) {
				Random random = new Random();
				r1 = random.nextDouble();
				r2 = random.nextDouble();
				double temp_beta = omega * weight_beta[swarmNum_index][i] + c1 * r1 * (pbest[swarmNum_index][i] - weight[swarmNum_index][i]) + c2 * r2 * (gbest[i] - weight[swarmNum_index][i]);
				weight_beta[swarmNum_index][i] = Math.min(Math.max(temp_beta, -(maxtid-0.1) * 0.2), (maxtid-0.1) * 0.2);
				weight[swarmNum_index][i] = Math.min(Math.max(weight[swarmNum_index][i] + weight_beta[swarmNum_index][i], 0.1), maxtid);
				sum+=weight[swarmNum_index][i];
			}
			if(sum>maxtid){
				for (int i = 0; i < maxtid; i++) {
					weight[swarmNum_index][i]*=maxtid/sum;
				}
			}
		}
	}

	public void initFactorMatrix() {
		U=new double[maxuid+1][rank+1];
		S=new double[maxsid+1][rank+1];
		U2=new double[maxuid+1][rank+1];
		S2=new double[maxsid+1][rank+1];
		T=new double[maxtid+1][rank+1];
		T2=new double[maxtid+1][rank+1];
		U3=new double[maxuid+1][rank+1];
		S3=new double[maxsid+1][rank+1];
		T3=new double[maxtid+1][rank+1];
		a=new double[maxuid+1];
		b=new double[maxsid+1];
		c=new double[maxtid+1];
		a_1=new double[maxtid+1][rank+1];
		b_1=new double[maxtid+1][rank+1];
		c_1=new double[maxtid+1][rank+1];
		count=new int[maxtid+1];
		w=new double[rank+1];
		m=new double[rank+1];
		square_=new double[maxtid+1];
		Mae_count=new double[maxtid+1];
		minMAE_=new double[maxtid+1];
		minRMSE_=new double[maxtid+1];
		for (int i = 1; i < maxtid+1; i++) {
			minMAE_[i]=Double.MAX_VALUE;
			minRMSE_[i]=Double.MAX_VALUE;
		}
		//小的随机值初始化
		Random ran=new Random();
		for(int i=0;i<maxuid+1;i++) {
			for(int r=0;r<rank+1;r++) {
				U[i][r]=initScale*ran.nextInt(scale)/scale;
				U2[i][r]=initScale*ran.nextInt(scale)/scale;
				U3[i][r]=U[i][r];
			}
			a[i]=initScale*ran.nextInt(scale)/scale;
		}
		for(int r=0;r<rank+1;r++) {
			w[r]=initScale*ran.nextInt(scale)/scale;
			m[r]=initScale*ran.nextInt(scale)/scale;
		}
		for(int j=0;j<maxsid+1;j++) {
			for(int r=0;r<rank+1;r++) {
				S[j][r]=initScale*ran.nextInt(scale)/scale;
				S2[j][r]=initScale*ran.nextInt(scale)/scale;
				S3[j][r]=S[j][r];
			}
			b[j]=initScale*ran.nextInt(scale)/scale;
		}
		for(int k=0;k<maxtid+1;k++) {
			for(int r=0;r<rank+1;r++) {
				T[k][r]=initScale*ran.nextInt(scale)/scale;
				T2[k][r]=initScale*ran.nextInt(scale)/scale;
				T3[k][r]=T[k][r];
				a_1[k][r]=initScale*ran.nextInt(scale)/scale;
				b_1[k][r]=initScale*ran.nextInt(scale)/scale;
				c_1[k][r]=initScale*ran.nextInt(scale)/scale;
			}
			c[k]=initScale*ran.nextInt(scale)/scale;

		}
	}
	
	//初始化辅助矩阵
	public void initAssistMatrix() {
		Uup=new double[maxuid+1][rank+1];
		Sup=new double[maxsid+1][rank+1];
		Tup=new double[maxtid+1][rank+1];
		Udown=new double[maxuid+1][rank+1];
		Sdown=new double[maxsid+1][rank+1];
		Tdown=new double[maxtid+1][rank+1];
		
		aup=new double[maxuid+1];
		bup=new double[maxsid+1];
		cup=new double[maxtid+1];
		adown=new double[maxuid+1];
		bdown=new double[maxsid+1];
		cdown=new double[maxtid+1];
		
		for(int i=0;i<maxuid+1;i++) {
			for(int r=0;r<rank+1;r++) {
				Uup[i][r]=0;
				Udown[i][r]=0;
			}
			aup[i]=0;
			adown[i]=0;
		}
		for(int j=0;j<maxsid+1;j++) {
			for(int r=0;r<rank+1;r++) {
				Sup[j][r]=0;
				Sdown[j][r]=0;
			}
			bup[j]=0;
			bdown[j]=0;
		}
		for(int k=0;k<maxtid+1;k++) {
			for(int r=0;r<rank+1;r++) {
				Tup[k][r]=0;
				Tdown[k][r]=0;
			}
			cup[k]=0;
			cdown[k]=0;
		}
	}
	
	//test assist intialization
	public void testAssist() {
		everyRoundMAE=new double[trainRound+1];
		everyRoundRMSE=new double[trainRound+1];
		RoundMae_1=new double[trainRound+1];
		RoundMae_2=new double[trainRound+1];
		RoundRmse_1=new double[trainRound+1];
		RoundRmse_2=new double[trainRound+1];
		everyRoundMAE[0]=everyRoundRMSE[0]=RoundMae_1[0]=RoundMae_2[0]=RoundRmse_1[0]=RoundRmse_2[0]=Double.MAX_VALUE;
		for(int i=1;i<trainRound+1;i++) {
			everyRoundMAE[i]=everyRoundRMSE[i]=0;
		}
	}
	
	//计算预测值
	public double getPre(int uid,int sid,int tid) {
		double pre=0;
		for(int r=1;r<rank+1;r++) {
			pre+=U[uid][r]*S[sid][r]*T[tid][r];
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return pre;
	}
	public double getPre(int uid,int sid) {
		double pre=0;
		for(int r=1;r<rank+1;r++) {
			pre+=U[uid][r]*S[sid][r];
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return pre;
	}
	public double getPre_duplicate(int uid,int sid,int tid) {
		double pre=0;
		if(tid==1) {
			for (int r = 1; r < rank + 1; r++) {
				pre += U[uid][r] * S[sid][r] * T[tid][r];
			}
		}
		if(tid==2){
			for (int r = 1; r < rank + 1; r++) {
				pre += U3[uid][r] * S3[sid][r] * T[tid][r];
			}
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return pre;
	}
	public double getPre_NoneLiner(int uid,int sid,int tid) {
		double pre=0;
		for(int r=1;r<rank+1;r++) {
		double temp=U[uid][r]*S[sid][r]*T[tid][r];
			pre+=temp*temp*a_1[tid][r]+temp*b_1[tid][r]+c_1[tid][r];
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return pre;
	}
	public double getPre_PSO(int uid,int sid,int tid) {
		double pre=0;
		for(int r=1;r<rank+1;r++) {
			pre+=U2[uid][r]*S2[sid][r]*T2[tid][r];
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return pre;
	}
	public double getPre_sigmoid(int uid,int sid,int tid) {
		double pre=0;
		for(int r=1;r<rank+1;r++) {
			double temp=U[uid][r]*S[sid][r]*T[tid][r]*w[r];
//			pre+=temp;
			pre+=Math.exp(temp)/(Math.exp(temp)+1)*m[r];
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return Math.exp(pre)/(Math.exp(pre)+1);
	}
	public double getPre_2(int uid,int sid,int tid) {
		double pre=0;
		if(tid==1) {
			for (int r = 1; r < rank + 1; r++) {
				pre += U[uid][r] * S[sid][r] * T[tid][r];
			}
		}
		if(tid==2) {
			for (int r = 1; r < rank + 1; r++) {
				pre += U[uid][r] * S2[sid][r] * T[tid][r];
			}
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return pre;
	}
	public double getPre_3(int uid,int sid,int tid) {
		double pre=0;
		if(tid==1) {
			for (int r = 1; r < rank + 1; r++) {
				pre += U[uid][r] * S[sid][r] * T[tid][r];
			}
		}
		if(tid==2) {
			for (int r = 1; r < rank + 1; r++) {
				pre += U2[uid][r] * S2[sid][r] * T[tid][r];
			}
		}
//		pre+=(a[uid]+b[sid]+c[tid]);
		return pre;
	}
}
