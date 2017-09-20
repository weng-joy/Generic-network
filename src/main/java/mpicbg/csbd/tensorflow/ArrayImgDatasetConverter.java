package mpicbg.csbd.tensorflow;

import java.lang.reflect.Array;
import java.util.Arrays;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.tensorflow.Tensor;

import mpicbg.csbd.normalize.Normalizer;

public class ArrayImgDatasetConverter implements DatasetConverter {

	@Override
	public Dataset tensorToDataset( final Tensor output_t, final DatasetTensorBridge bridge ) {

		if ( output_t != null ) {
			/*
			 * create 5D array from output tensor, unused dimensions will
			 * have size 1
			 */
			final float[][][][][] outputarr = bridge.createTFArray5D( output_t );

			for ( int i = 0; i < output_t.numDimensions(); i++ ) {
				System.out.println( "output dim " + i + ": " + output_t.shape()[ i ] );
			}

			if ( output_t.numDimensions() == bridge.getInitialInputTensorShape().numDimensions() - 1 ) {
				//model reduces dim by 1
				//assume z gets reduced -> move it to front and ignore first dimension
				/*
				 * model reduces dim by 1
				 * assume z gets reduced -> move it to front and ignore
				 * first dimension
				 */
				System.out.println( "model reduces dimension, z dimension reduction assumed" );
				bridge.removeZFromMapping();
			}

			// .. :-/
			if ( output_t.numDimensions() == 5 ) {
				output_t.copyTo( outputarr );
			} else {
				if ( output_t.numDimensions() == 4 ) {
					output_t.copyTo( outputarr[ 0 ] );
				} else {
					if ( output_t.numDimensions() == 3 ) {
						output_t.copyTo( outputarr[ 0 ][ 0 ] );
					}
				}
			}

			return arrayToDataset( outputarr, output_t.shape(), bridge );

		}

		return null;
	}

	protected Dataset arrayToDataset(
			final float[][][][][] outputarr,
			final long[] shape,
			final DatasetTensorBridge bridge ) {

		final Dataset img_out = bridge.createDatasetFromTFDims( shape );

		//write ouput dataset and undo normalization

		final Cursor< RealType< ? > > cursor = img_out.localizingCursor();
		while ( cursor.hasNext() ) {
			final int[] pos = { 0, 0, 0, 0, 0 };
			final RealType< ? > val = cursor.next();
			for ( int i = 0; i < pos.length; i++ ) {
				final int imgIndex = bridge.getDatasetDimIndexByTFIndex( i );
				if ( imgIndex >= 0 ) {
					pos[ i ] = cursor.getIntPosition( imgIndex );
				}
			}
//			System.out.println("pos " + pos[0] + " " + pos[1] + " " + pos[2] + " " + pos[3] + " " + pos[4]);
			val.setReal( outputarr[ pos[ 0 ] ][ pos[ 1 ] ][ pos[ 2 ] ][ pos[ 3 ] ][ pos[ 4 ] ] );

		}

		return img_out;

	}

	@Override
	public Tensor datasetToTensor(
			final Dataset image,
			final DatasetTensorBridge bridge,
			final Normalizer normalizer ) {
		
		bridge.updatedMapping();
		
		long[] finalDim = bridge.getFinalInputTensorShape();
		final ArrayImg<FloatType, FloatArray> dest = ArrayImgs.floats(finalDim);
		final Cursor<FloatType> destCursor = dest.localizingCursor();
		RandomAccess< RealType< ? > > source = image.randomAccess();
		
		System.out.println("datasetToArray");
		System.out.print("dest dimension: ");
		for(long dim : finalDim){
			System.out.print(dim + " ");
		}
		System.out.println();
		System.out.print("source dimension: ");
		for(int i = 0; i < image.numDimensions(); i++){
			System.out.print(image.dimension(i) + " ");
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
				float dval = source.get().getRealFloat();
				if(normalizer.isActive()){
					dval = normalizer.normalize( dval );
				}
//				System.out.println(" val: " + dval);
//				System.out.println("pos " + pos[0] + " " + pos[1] + " " + pos[2] + " " + pos[3] + " " + pos[4]);
				setValue(multiDimArray, new Float(dval), posdest);
				destCursor.get().setReal(dval);
			}
			
//			final ImageJ ij = new ImageJ();
//			ij.ui().show(dest);
			
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
}
