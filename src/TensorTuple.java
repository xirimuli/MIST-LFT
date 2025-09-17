package LFT.cn.edu.swu;

public class TensorTuple {
	//position index and value
	public int uid=0;
	public int sid=0;
	public int tid=0;
	
	public double value=0;
	public double valueHat=0;

	@Override
	public String toString() {
		return "TensorTuple{" +
				"uid=" + uid +
				", sid=" + sid +
				", tid=" + tid +
				", value=" + value +
				", valueHat=" + valueHat +
				'}';
	}
}
