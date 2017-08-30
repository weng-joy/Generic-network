/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package mpicbg.csbd;

import com.google.protobuf.InvalidProtocolBufferException;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

import javax.swing.JOptionPane;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.Cancelable;
import org.scijava.ItemIO;
import org.scijava.ItemVisibility;
import org.scijava.command.Command;
import org.scijava.command.Previewable;
import org.scijava.io.location.FileLocation;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;
import org.scijava.widget.Button;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlowException;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;

/**
 */
@Plugin(type = Command.class, menuPath = "Plugins>CSBDeep", headless = true)
public class CSBDeep<T extends RealType<T>> implements Command, Previewable, Cancelable {
	
	@Parameter(visibility = ItemVisibility.MESSAGE)
	private String header = "This command removes noise from your images.";

    @Parameter(label = "input data", type = ItemIO.INPUT, callback = "imageChanged", initializer = "imageChanged")
    private Dataset input;
    
    @Parameter(label = "Normalize image")
	private boolean normalizeInput = true;
    
    @Parameter(label = "Import model", callback = "modelChanged", initializer = "modelChanged")
    private File modelfile;
    
    @Parameter(label = "Input node name", callback = "inputNodeNameChanged", initializer = "inputNodeNameChanged")
    private String inputNodeName = "input";
    
    @Parameter(label = "Output node name", persist = false)
    private String outputNodeName = "output";
    
    @Parameter(label = "Adjust image <-> tensorflow mapping", callback = "openTFMappingDialog")
	private Button changeTFMapping;
    
    @Parameter
	private TensorFlowService tensorFlowService;
    
    @Parameter
	private mpicbg.csbd.TensorFlowService tensorFlowService2;
    
    @Parameter
	private LogService log;

    @Parameter
    private UIService uiService;

    @Parameter
    private OpService opService;
    
    @Parameter(type = ItemIO.OUTPUT)
    private Dataset outputImage;
    
    @Parameter
    private double percentileBottom = 0.1;
    @Parameter
    private double percentileTop = 0.9;
    @Parameter
    private double min = 0;
    @Parameter
    private double max = 100;
    private Graph graph;
    private SavedModelBundle model;
    private SignatureDef sig;
    private DatasetTensorBridge bridge;
    private boolean hasSavedModel = true;
    
	// Same as
	// tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
	// in Python. Perhaps this should be an exported constant in TensorFlow's Java
	// API.
	private static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default";
    
    public CSBDeep(){
//    	modelChanged();
    }
    	
	@Override
	public void preview() {
//		imageChanged();
//		modelChanged();
	}
	
	/*
	 * model can be imported via graphdef or savedmodel
	 */
	protected boolean loadGraph(){
		
//		System.out.println("loadGraph");
		
		if(modelfile == null){
			System.out.println("Cannot load graph from null File");
			return false;
		}
		
		final FileLocation source = new FileLocation(modelfile);
		hasSavedModel = true;
		try {
			model = tensorFlowService.loadModel(source, modelfile.getName());
		} catch (TensorFlowException | IOException e) {
			try {
				graph = tensorFlowService2.loadGraph(modelfile);
//				graph = tensorFlowService.loadGraph(source, "", "");
				hasSavedModel = false;
			} catch (final IOException e2) {
				e2.printStackTrace();
				return false;
			}
		}
		return true;
	}
	
	protected boolean loadModelInputShape(final String inputName){
		
//		System.out.println("loadModelInputShape");
		
		if(getGraph() != null){
			final Operation input_op = getGraph().operation(inputName);
			if(input_op != null){
				bridge.setInputTensorShape(input_op.output(0).shape());
				return true;			
			}
			System.out.println("input node with name " + inputName + " not found");			
		}
		return false;
	}
	
	/*
	 * model can be imported via graphdef or savedmodel, depending on that the execution graph has different origins
	 */
	protected Graph getGraph(){
		if(hasSavedModel && (model == null)){
			return null;
		}
		return hasSavedModel ? model.graph() : graph;
	}
    
    /** Executed whenever the {@link #input} parameter changes. */
	protected void imageChanged() {
		
//		System.out.println("imageChanged");
		
		if(input != null) {
			bridge = new DatasetTensorBridge(input);
		}
		
	}
	
    /** Executed whenever the {@link #modelfile} parameter changes. */
	protected void modelChanged() {
		
//		System.out.println("modelChanged");
		
		imageChanged();
		if(loadGraph()){
			
			if(hasSavedModel){
				// Extract names from the model signature.
				// The strings "input", "probabilities" and "patches" are meant to be
				// in sync with the model exporter (export_saved_model()) in Python.
				try {
					sig = MetaGraphDef.parseFrom(model.metaGraphDef())
						.getSignatureDefOrThrow(DEFAULT_SERVING_SIGNATURE_DEF_KEY);
				} catch (final InvalidProtocolBufferException e) {
//					e.printStackTrace();
					hasSavedModel = false;
				}
				if(sig != null && sig.isInitialized()){
					if(sig.getInputsCount() > 0){
						inputNodeName = sig.getInputsMap().keySet().iterator().next();					
					}
					if(sig.getOutputsCount() > 0){
						outputNodeName = sig.getOutputsMap().keySet().iterator().next();					
					}
				}				
			}

			inputNodeNameChanged();
		}
	}
	
	/** Executed whenever the {@link #inputNodeName} parameter changes. */
	protected void inputNodeNameChanged() {
		
//		System.out.println("inputNodeNameChanged");
		
		loadModelInputShape(inputNodeName);
		
		if(bridge.getInitialInputTensorShape() != null){
			if(!bridge.isMappingInitialized()){
				bridge.setMappingDefaults();
			}
		}
	}
	
	protected void openTFMappingDialog() {
		
		imageChanged();
		
		if(bridge.getInitialInputTensorShape() == null){
			modelChanged();
		}
		
		MappingDialog.create(bridge, sig);
	}
	private static long[] reversedDims(long[] inputdims) {
		final long[] dims = new long[inputdims.length];
		for (int d = 0; d < dims.length; d++) {
			dims[dims.length - d - 1] = inputdims[d];
		}
		return dims;
	}

	@Override
    public void run() {
		
//		System.out.println("run");
		
//		Dataset input_norm = input.duplicate();
		
		if(graph == null){
			modelChanged();
		}

		try (
			final Tensor image = datasetToArray(input);
		)
		{
			outputImage = executeGraph(getGraph(), image);	
			uiService.show(outputImage);
		}
		
//		uiService.show(arrayToDataset(datasetToArray(input)));
		
    }
	
	public List<Double> getPercentileValues(Dataset d, List<Double> percentiles){
		List<Double> percentileValues = null;
		
		
		return percentileValues;
	}
	
	/*
	 * create 5D array from dataset (unused dimensions get size 1)
	 */
	private Tensor datasetToArray(final Dataset d) {
		
		bridge.updatedMapping();
		
		long[] finalDim = bridge.getFinalInputTensorShape();
		final ArrayImg<FloatType, FloatArray> dest = ArrayImgs.floats(finalDim);
		final Cursor<FloatType> destCursor = dest.localizingCursor();
		RandomAccess<RealType<?>> source = d.randomAccess();
		
		System.out.println("datasetToArray");
		System.out.print("dest dimension: ");
		for(long dim : finalDim){
			System.out.print(dim + " ");
		}
		System.out.println();
		System.out.print("source dimension: ");
		for(int i = 0; i < d.numDimensions(); i++){
			System.out.print(d.dimension(i) + " ");
		}
		System.out.println();

//		if (min == min && max == max) {
//			// Normalize the data.
//			final double range = max - min;
//			while (destCursor.hasNext()) {
//				destCursor.fwd();
//				source.setPosition(destCursor);
//				final double value = (source.get().getRealDouble() - min) / range;
//				destCursor.get().setReal(value);
//			}
//		}
//		else {
			// Do not perform normalization.
		
			int[] lookup = new int[source.numDimensions()];
			for(int i = 0; i < lookup.length; i++){
				lookup[i] = bridge.getTFIndexByDatasetDimIndex(i);
			}
			System.out.println();
			System.out.print("lookup: ");
			for(int i = 0; i < lookup.length; i++){
				System.out.print(lookup[i] + " ");
			}
			System.out.println();
			
			System.out.println();
			System.out.print("dest: ");
			for(int i = 0; i < dest.numDimensions(); i++){
				System.out.print(dest.dimension(i) + " ");
			}
			System.out.println();

			int[] finaldimi = new int[finalDim.length];
			for(int i = 0; i < finalDim.length; i++){
				finaldimi[i] = (int)finalDim[i];
			}
			Object multiDimArray = Array.newInstance(Float.class, finaldimi);
			
			while( destCursor.hasNext() )
			{
				
				int[] posdest = new int[dest.numDimensions()];
//				System.out.print("posdest: ");
				for(int i = 0; i < posdest.length; i++){
					posdest[i] = Math.max(0,destCursor.getIntPosition(i));
//					System.out.print(posdest[i] + " ");
//					System.out.print(" (dimension " + dest.dimension(i) + ") ");
				}
//				System.out.println();
				
				destCursor.fwd();
				int[] pos = new int[source.numDimensions()];
//				System.out.print("pos: ");
				for(int i = 0; i < pos.length; i++){
					pos[i] = destCursor.getIntPosition(lookup[i]);
//					System.out.print(pos[i] + " ");
				}
				source.setPosition(pos);
				final double dval = source.get().getRealDouble();
//				System.out.println(" val: " + dval);
//				System.out.println("pos " + pos[0] + " " + pos[1] + " " + pos[2] + " " + pos[3] + " " + pos[4]);
				setValue(multiDimArray, new Float(dval), posdest);
				destCursor.get().setReal(dval);
			}
			
			uiService.show(dest);
			
//		}
			
//			float[] res = dest.update(null).getCurrentStorageArray();
//			try {
//				write("/home/random/test-res.txt", res);
//			} catch (IOException exc) {
//				// TODO Auto-generated catch block
//				exc.printStackTrace();
//			}
			
//			FloatBuffer buf = FloatBuffer.wrap(res);
			return Tensor.create(multiDimArray);
		
	}
	
	public static int[] tail(int[] arr) {
        return Arrays.copyOfRange(arr, 1, arr.length);
    }

    public static void setValue(Object array, float value, int... indices) {
        if (indices.length == 1)
            ((Float[]) array)[indices[0]] = value;
        else
            setValue(Array.get(array, indices[0]), value, tail(indices));
    }
	
	public static void write (String filename, float[]x) throws IOException{
		  BufferedWriter outputWriter = new BufferedWriter(new FileWriter(filename));
		  outputWriter.write(Arrays.toString(x));
		  outputWriter.flush();  
		  outputWriter.close();  
		}
	
	/*
	 * runs graph on input tensor
	 * converts result tensor to dataset 
	 */
	private Dataset executeGraph(final Graph g, final Tensor image)
		{	
		
		System.out.println("executeInceptionGraph");
		
		try (
				Session s = new Session(g);
		) {
			
//			int size = s.runner().feed(inputNodeName, image).fetch(outputNodeName).run().size();
//			System.out.println("output array size: " + size);
			
			Tensor output_t = null;
			
			/*
			 * check if keras_learning_phase node has to be set
			 */
			if(graph.operation("dropout_1/keras_learning_phase") != null){
				final Tensor learning_phase = Tensor.create(false);
				try{
					/*
					 * execute graph
					 */
					final Tensor output_t2 = s.runner().feed(inputNodeName, image).feed("dropout_1/keras_learning_phase", learning_phase).fetch(outputNodeName).run().get(0);
					output_t = output_t2;
				}
				catch(final Exception e){
					e.printStackTrace();
				}
			}else{
				try{
					/*
					 * execute graph
					 */
					final Tensor output_t2 = s.runner().feed(inputNodeName, image).fetch(outputNodeName).run().get(0);
					output_t = output_t2;
				}
				catch(final Exception e){
					e.printStackTrace();
				}
			}
			
			if(output_t != null){
				
				System.out.println("Output tensor with " + output_t.numDimensions() + " dimensions");
				
				if(output_t.numDimensions() == 0){
					showError("Output tensor has no dimensions");
					return null;
				}
				
//				FloatBuffer resvals = null;
//				output_t.writeTo(resvals);
//				
//				final ArrayImg<FloatType, FloatArray> resimg = ArrayImgs.floats(resvals.array(), output_t.shape());
				
				
				
				
				/*
				 * create 5D array from output tensor, unused dimensions will have size 1
				 */
				final float[][][][][] outputarr = bridge.createTFArray5D(output_t);
				
				for(int i = 0; i < output_t.numDimensions(); i++){
					System.out.println("output dim " + i + ": " + output_t.shape()[i]);
				}
				
				if(output_t.numDimensions() -1 == bridge.getInitialInputTensorShape().numDimensions()){
					/*
					 * model reduces dim by 1
					 * assume z gets reduced -> move it to front and ignore first dimension
					 */
					System.out.println("model reduces dimension, z dimension reduction assumed");
					bridge.moveZMappingToFront();
				}
				
				// .. :-/
				if(output_t.numDimensions() == 5){
					output_t.copyTo(outputarr);
				}else{
					if(output_t.numDimensions() == 4){
						output_t.copyTo(outputarr[0]);					
					}else{
						if(output_t.numDimensions() == 3){
							output_t.copyTo(outputarr[0][0]);
						}
					}
				}
				
				return arrayToDataset(outputarr, output_t.shape());	
			}
			return null;
			
			
		}
		catch (final Exception e) {
			System.out.println("could not create output dataset");
			e.printStackTrace();
		}
		return null;
	}
	
	private Dataset arrayToDataset(final float[][][][][] outputarr, final long[] shape){
		
		final Dataset img_out = bridge.createDatasetFromTFDims(shape);
		
		//write ouput dataset and undo normalization
		
		final Cursor<T> cursor = (Cursor<T>) img_out.localizingCursor();
		while( cursor.hasNext() )
		{
			final int[] pos = {0,0,0,0,0};
			final T val = cursor.next();
			for(int i = 0; i < pos.length; i++){
				final int imgIndex = bridge.getDatasetDimIndexByTFIndex(i);
				if(imgIndex >= 0){
					pos[i] = cursor.getIntPosition(imgIndex);
				}
			}
//			System.out.println("pos " + pos[0] + " " + pos[1] + " " + pos[2] + " " + pos[3] + " " + pos[4]);
			val.setReal(outputarr[pos[0]][pos[1]][pos[2]][pos[3]][pos[4]]);
			
		}

		return img_out;
		
	}

    /**
     * This main function serves for development purposes.
     * It allows you to run the plugin immediately out of
     * your integrated development environment (IDE).
     *
     * @param args whatever, it's ignored
     * @throws Exception
     */
    public static void main(final String... args) throws Exception {
        // create the ImageJ application context with all available services
        final ImageJ ij = new ImageJ();

        // ask the user for a file to open
//        final File file = ij.ui().chooseFile(null, "open");
        final File file = new File("/home/random/input.png");
        
        if(file.exists()){
            // load the dataset
            final Dataset dataset = ij.scifio().datasetIO().open(file.getAbsolutePath());

            // show the image
            ij.ui().show(dataset);

            // invoke the plugin
            ij.command().run(CSBDeep.class, true);
        }    	

    }
    
    public void showError(final String errorMsg) {
    	JOptionPane.showMessageDialog(null, errorMsg, "Error",
                JOptionPane.ERROR_MESSAGE);
    }

	@Override
	public void cancel() {
		// TODO Auto-generated method stub
	}

	@Override
	public boolean isCanceled() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void cancel(final String reason) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getCancelReason() {
		// TODO Auto-generated method stub
		return null;
	}

}
