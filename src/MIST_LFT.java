package LFT.cn.edu.swu;

import java.io.IOException;

public class MIST_LFT extends InitTensor {

    public double yita = 0;
    public double lambda = 0, lambda2 = 0, lambda3 = 0;
    public double z1 = 0, z2 = 0;
    public double lambda_b = 0;
    public double errgap = 1e-5;
    public int tr = 0;
    public int threshold = 5;
    public boolean flag1 = true, flag2 = true;
    public double[] weight = new double[maxtid];

    public MIST_LFT(String trainFilePath, String testFilePath, String seperator) {
        super(trainFilePath, testFilePath, seperator);
    }

    //train via SGDs
    public void train() throws IOException {
        int round = 1;
        double[] sumLossCumulative = new double[maxtid + 1];
        for (; round < trainRound; round++) {
            //Non-spatio-temporal correlation constraints part
            for (TensorTuple e : trainData) {
                e.valueHat = getPre(e.uid, e.sid, e.tid);
                double rateing = weight[e.tid - 1] * (e.value - e.valueHat);
                for (int r = 1; r < rank + 1; r++) {
                    U[e.uid][r] = U[e.uid][r] + yita * (
                            rateing * S[e.sid][r] * T[e.tid][r]
                                    - lambda * U[e.uid][r]
                    );
                    S[e.sid][r] = S[e.sid][r] + yita * (
                            rateing * U[e.uid][r] * T[e.tid][r]
                                    - lambda2 * S[e.sid][r]
                    );
                    T[e.tid][r] = T[e.tid][r] + yita * (
                            rateing * U[e.uid][r] * S[e.sid][r]
                                    - lambda3 * T[e.tid][r]
                    );
                }
            }
            //Spatio-temporal correlation constraints part
            for (int k = 1; k <= maxtid; k++) {
                for (int i = 0; i < maxuid; i++) {
                    double l = 0;
                    for (int j = 0; j < maxuid; j++) {
                        l += L[j][i];
                    }
                    for (int j = 0; j < maxsid; j++) {
                        double lust = 0;
                        for (int r = 1; r < rank + 1; r++) {
                            for (int z = 0; z < maxuid; z++) {
                                lust += L[i][z] * U[z][r] * S[j][r] * T[k][r];
                            }
                        }
                        double usdt = 0;
                        double[] lu = new double[rank + 1];
                        for (int r = 1; r < rank + 1; r++) {
                            for (int u = 0; u < maxuid; u++) {
                                lu[r] += L[i][u] * U[u][r];
                            }
                        }
                        if (j == 0) {
                            for (int r = 1; r < rank + 1; r++) {
                                {
                                    usdt += U[i][r] * (S[j + 1][r] - S[j][r]) * T[k][r];
                                }
                            }
                            for (int r = 1; r < rank + 1; r++) {
                                U[i][r] = U[i][r] + yita * (
                                        -z1 * lust * l * S[j][r] * T[k][r]
                                                - z2 * usdt * (S[j + 1][r] - S[j][r]) * T[k][r]
                                );
                                S[j][r] = S[j][r] + yita * (
                                        -z1 * lust * lu[r] * T[k][r]
                                                + z2 * usdt * U[i][r] * T[k][r]
                                );
                                T[k][r] = T[k][r] + yita * (
                                        -z1 * lust * lu[r] * S[j][r]
                                                - z2 * usdt * (S[j + 1][r] - S[j][r]) * U[i][r]
                                );
                            }
                        } else if (j == maxsid - 1) {
                            for (int r = 1; r < rank + 1; r++) {
                                {
                                    usdt += U[i][r] * (S[j][r] - S[j - 1][r]) * T[k][r];
                                }
                            }
                            for (int r = 1; r < rank + 1; r++) {
                                U[i][r] = U[i][r] + yita * (
                                        -z1 * lust * l * S[j][r] * T[k][r]
                                );
                                S[j][r] = S[j][r] + yita * (
                                        -z1 * lust * lu[r] * T[k][r]
//												-z2*usdt*U[i][r]*T[k][r]
                                );
                                T[k][r] = T[k][r] + yita * (
                                        -z1 * lust * lu[r] * S[j][r]
                                );
                            }
                        } else {
                            double usdt2 = 0;
                            for (int r = 1; r < rank + 1; r++) {
                                {
                                    usdt += U[i][r] * (S[j + 1][r] - S[j][r]) * T[k][r];
                                    usdt2 += U[i][r] * (S[j][r] - S[j - 1][r]) * T[k][r];
                                }
                            }
                            for (int r = 1; r < rank + 1; r++) {
                                U[i][r] = U[i][r] + yita * (
                                        -z1 * lust * l * S[j][r] * T[k][r]
                                                - z2 * usdt * (S[j + 1][r] - S[j][r]) * T[k][r]
                                );
                                S[j][r] = S[j][r] + yita * (
                                        -z1 * lust * lu[r] * T[k][r]
                                                - z2 * (usdt2 - usdt) * U[i][r] * T[k][r]
                                );
                                T[k][r] = T[k][r] + yita * (
                                        -z1 * lust * lu[r] * S[j][r]
                                                - z2 * usdt * (S[j + 1][r] - S[j][r]) * U[i][r]
                                );
                            }
                        }
                    }
                }
            }
            //adaptively updating the alpha
            double[] sumLoss = new double[maxtid + 1];
            for (TensorTuple e : trainData) {
                e.valueHat = getPre(e.uid, e.sid, e.tid);
                double v1 = e.value, v2 = e.valueHat;
                sumLoss[e.tid] += Math.abs(v1 - v2);
            }
            double sum = 0;
            double math_sum = 0;
            for (int i = 1; i <= maxtid; i++) {
                sumLossCumulative[i] += sumLoss[i];
                sum += sumLossCumulative[i];
            }
            for (int i = 1; i <= maxtid; i++) {
                math_sum += Math.exp(-(sumLossCumulative[i] / sum));
            }
            for (int i = 1; i <= maxtid; i++) {
                weight[i - 1] = Math.exp(-(sumLossCumulative[i] / sum)) / math_sum;
                weight[i - 1] = weight[i - 1] * maxtid;
            }
            ////////////////////////////////////////////////////////////

            //test
            double square = 0, absCount = 0;
            boolean flag = false;
            StringBuffer str = new StringBuffer();
            for (int i = 1; i <= maxtid; i++) {
                square_[i] = 0;
                Mae_count[i] = 0;
                count[i] = 0;
            }
            for (TensorTuple e : testData) {
                e.valueHat = getPre(e.uid, e.sid, e.tid);
                if (e.tid == 1) {
                    e.valueHat = (e.valueHat - 2) / 498 * (122.153 - 0.016) + 0.016;
                }
                if (e.tid == 2) {
                    e.valueHat = (e.valueHat - 2) / 498 * (8983.13 - 0.537709) + 0.537709;
                }
                if (e.tid == 3) {
                    e.valueHat = (e.valueHat - 2) / 498 * (2.71196 - 2.33827) + 2.33827;
                }
                double v1 = e.value, v2 = e.valueHat;
                square_[e.tid] += Math.pow(v1 - v2, 2);
                Mae_count[e.tid] += Math.abs(v1 - v2);
                count[e.tid]++;
                square += Math.pow(e.value - e.valueHat, 2);
                absCount += Math.abs(e.value - e.valueHat);
            }
            for (int i = 1; i <= maxtid; i++) {
                square_[i] = Math.sqrt(square_[i] / count[i]);
                Mae_count[i] = Mae_count[i] / count[i];
                if (minMAE_[i] > Mae_count[i]) {
//					flag=true;
                    minMAE_[i] = Mae_count[i];
//					tr=0;

                    tr = 0;
                    flag = true;

                }
                if (minRMSE_[i] > square_[i]) {

                    minRMSE_[i] = square_[i];

                    tr = 0;
                    flag = true;

                }
                str.append("	t" + String.valueOf(i) + "_Rmse:" + square_[i] + "	t" + String.valueOf(i) + "_MAE:" + Mae_count[i]);
            }
            everyRoundRMSE[round] = Math.sqrt(square / testDataCount);
            everyRoundMAE[round] = absCount / testDataCount;
            minMAE = Math.min(minMAE, everyRoundMAE[round]);
            minRMSE = Math.min(minRMSE, everyRoundRMSE[round]);
            if (!flag) {
                tr++;
                if (tr == threshold) {
                    break;
                }
            }
        }
        StringBuffer str = new StringBuffer();
        for (int i = 1; i <= maxtid; i++) {
            str.append("	t" + String.valueOf(i) + "_MINRmse:" + minRMSE_[i] + "	t" + String.valueOf(i) + "_MINMAE:" + minMAE_[i]);
        }
        System.out.println("r:" + rank + "	round:   " + round + str);
    }

    public static void main(String[] args) throws IOException {
        String path = "Data\\";
        {
            {

                double k = 0.001;
                MIST_LFT bn = new MIST_LFT(path + "BeijingAirQuality\\" + String.valueOf(k) + "_sample_rating\\train.txt",
                        path + "BeijingAirQuality\\" + String.valueOf(k) + "_sample_rating\\test.txt", "::");
                //configuration
                bn.rank = 20;
                bn.lambda = 0.0001 * 20;//0.002 ртоб
                bn.lambda2 = 0.0001 * 20;
                bn.lambda3 = 0.0001 * 10;
                bn.lambda_b = 0.01;
                bn.z1 = 0.0004;
                bn.z2 = 0.002;
                bn.threshold = 30;
                bn.errgap = 1e-10;
                bn.tr = 0;
                bn.yita = 0.000004 * 2;//0.00038
                bn.initScale = 0.4;
                double[] a = {1, 1, 1};
                bn.weight = a;
                //init
                bn.initData(bn.trainFilePath, bn.trainData, 1);
                bn.initLaplace(path + "L.txt");
                bn.initD();
                bn.initData(bn.testFilePath, bn.testData, 2);
                bn.initFactorMatrix();
                bn.testAssist();
                bn.train();
                System.out.println("K:" + "---------------");

            }
        }

    }
}
