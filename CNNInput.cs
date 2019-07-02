using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// We are going to launch Tensorflow in a different thread
using System.Threading;

// Include Tensorflow, required to run the Keras background we imported
using TensorFlow;

public class CNNInput : MonoBehaviour {
	// Graph with the Model of the CNN loaded
	public TextAsset ModelGraph;

	// Dropdown menu to switch between the different types of neural network implemented
	public enum NNType
	{
		VectorDataNN,
		GrayScaleImageCNN,
		LSTM
	};

	public NNType nNType;
	
	// Number of rows in the input shape
	public float NumRows = 59f;
	// Number of columns in the input shape
	public float NumCols = 112f;

	// Minimum horizontal value posible for a player's position
	public float StartingX = -24f;
	// Minimum vertical value posible for a player's position
	public float StartingY = -15f;
	// Width, in units, of the field
	public float Width = 54f;
	// Height, in units, of the field
	public float Height = 27f;
	
	// Ball of the game
	GameObject ball;
	// The player, and because of that, the teammate of this agent
	GameObject Player;
	// First of the members of enemy team
	GameObject Enemy1;
	// Second of the members of enemy team
	GameObject Enemy2;

	/* Inputs of the model, they vary depending on the NN type selected */
	private float[,,,] InputImage;

	private float[,] InputVector;

	private float[,,] InputVectorLSTM;
	
	// Index of the current frame we are getting data from
	private int CurrentFrameToGetData = 0;

	// Number of frames we are providing the neural network each time
	public int NFrames = 2;
	
	// Current direction infered by the network, in grades (Not movement is not supported)
	private float DirectionInGrades = -1f;

	// Thread for executing the Tensorflow model
	Thread ModelThread;
	// Temporary variable for storing the output of the model, until the thread finishes
	private float TemporaryDirectionInGrades = -1f;

	// Graph of the model
	TFGraph graph = new TFGraph();
	// Current session of the graph
	TFSession session;

	// Show graph in log, for debug purposes
	[SerializeField]
	public bool showGraph = false;
	
	// Should the data be normalized?
	[SerializeField]
	public bool normalizeData = true;

	// Use this for initialization
	void Start () {
		// We need this to be able to build in Android with TensorflowSharp
		#if UNITY_ANDROID && !UNITY_EDITOR
			TensorFlowSharp.Android.NativeBinding.Init();
		#endif

		VectorDataInitialization();

		LoadModel();

		if(showGraph)
		{
			OutputModel();
		}

		// Try to find the other gameobjects, for building the data properly
		TryToFindMembers();

		// Initialize the thread we are going to use to evaluate the current state of the game
		// ModelThread = new Thread( Evaluate );
		// ModelThread.Priority = System.Threading.ThreadPriority.Lowest;
		// ModelThread.Start();
	}

	private void VectorDataInitialization()
	{
		switch(nNType)
		{
			case NNType.LSTM:
				InputVectorLSTM = new float[1, NFrames, 10];
				break;
			case NNType.GrayScaleImageCNN:
				// Create the Input Image of reference, and add the player and other agents
				InputImage = new float[1000,(int)NumRows, (int)NumCols, 1];
				CreateInputShape();
				FillSquareGivenPosition(32, 57, 0.5f);
				FillSquareGivenPosition(50, 62, 0.75f);
				FillSquareGivenPosition(48, 91, 0f);
				FillSquareGivenPosition(33, 100, 0.25f);
				FillSquareGivenPosition(49, 99, 0.25f);
				break;
			case NNType.VectorDataNN:
				InputVector = new float[1, 10];
				break;
			default:
				break;
		}
	}

	// Try to find the remaining gameobjects we need and still don't have
	private void TryToFindMembers()
	{
		// Check if we already have the ball
		if(!ball)
		{
			// Try to find it
			ball = GameObject.Find("ballerino");
		}
		
		// Check if we already have the first enemy
		if(!Enemy1)
		{
			// Try to find it
			Enemy1 = GameObject.Find("bot_new(Clone)0");
		}
		
		// Check if we already have the second enemy
		if(!Enemy2)
		{
			// Try to find it
			Enemy2 = GameObject.Find("bot_new(Clone)1");
		}

		// Check if we already have the player
		if(!Player)
		{
			// Try to find it
			Player = GameObject.Find("player(Clone)");
		}
	}

	// Checks if we have a reference to all the other members we need for the grayscale image
	private bool HasAllMembers()
	{
		if(ball && Enemy1 && Enemy2 && Player)
		{
			return true;
		}
		return false;
	}
	
	// Update is called once per frame
	void Update()
	{
		// First, check to see if we have all the references needed from the other agents in the game
		if(!HasAllMembers())
		{
			TryToFindMembers();
		}
		else
		{
			// Then, check if the Thread with the Tensorflow Model is still running
			// if( !ModelThread.IsAlive )
			// {
				// If not, create a new input image with the data from the current moment
				if(nNType == NNType.GrayScaleImageCNN)
				{
					CreateImageFromData();
				}
				if(nNType == NNType.LSTM || nNType == NNType.VectorDataNN)
				{
					CreateVectorFromData();
				}
				
				if(nNType == NNType.LSTM)
				{
					CurrentFrameToGetData++;
					if(CurrentFrameToGetData == NFrames)
					{
						CurrentFrameToGetData = 0;
						Evaluate();
					}
				}
				else
				{
					Evaluate();
				}
			
				// And also, save the new inference of the model, so the agent can access this
				DirectionInGrades = TemporaryDirectionInGrades;
				// Finally, start a new thread to obtain a new value from the model
				// ModelThread = new Thread( Evaluate );
				// ModelThread.Priority = System.Threading.ThreadPriority.Lowest;
				// ModelThread.Start();
			// }
		}
	}

	// Fill the Input Matrix with empty data to test
	void CreateInputShape()
	{
		for(int i = 0; i < NumRows; i++)
		{
			for(int j = 0; j < NumCols; j++)
			{
				// Fill everything with 0
				InputImage[0, i, j, 0] = 0f;
			}
		}
	}

	// Discretizes the X coordinate to the format required by the CNN
	private int TransformXCoordinate(float X)
	{
		int Answer = (int)((NumCols-2f) * (X - StartingX) / Width);
		return Answer;
	}

	// Discretizes the Y coordinate to the format required by the CNN
	private int TransformYCoordinate(float Y)
	{
		int Answer = (int)((NumRows-2f) * (Y - StartingY) / Height);
		return Answer;
	}

	// Create a new Input Image from the actual data of the game
	void CreateImageFromData()
	{
		// Restart the image
		CreateInputShape();

		// Obtain current position
		Vector3 position = gameObject.transform.position;
		// Transform the position in our image format
		int X = TransformXCoordinate(position.x);
		int Y = TransformYCoordinate(position.y);

		// Add the player to the image
		FillSquareGivenPosition(Y, X, 0.5f);

		// Debug.Log("Position player: " + X + " - " + Y);

		// Check if we have the ball
		if(ball)
		{
			// Obtain ball position
			Vector3 bPosition = ball.transform.position;

			// Transform the position in our image format
			int bX = TransformXCoordinate(bPosition.x);
			int bY = TransformYCoordinate(bPosition.y);

			// Add the ball to the image
			FillSquareGivenPosition(bY, bX, 0f);
		}

		// Check if we have the player (from this perspective, is his teammate)
		if(Player)
		{
			// Obtain the player's position
			Vector3 pPosition = Player.transform.position;

			// Transform the position in our image format
			int pX = TransformXCoordinate(pPosition.x);
			int pY = TransformYCoordinate(pPosition.y);

			// Add the player to the image
			FillSquareGivenPosition(pY, pX, 0.75f);
		}

		// Check if we have the enemy 1
		if(Enemy1)
		{
			// Obtain the enemy's position
			Vector3 e1Position = Enemy1.transform.position;

			// Transform the position in our image format
			int e1X = TransformXCoordinate(e1Position.x);
			int e1Y = TransformYCoordinate(e1Position.y);

			// Add the enemy to the image
			FillSquareGivenPosition(e1Y, e1X, 0.25f);
		}

		// Check if we have the enemy 2
		if(Enemy2)
		{
			// Obtain the enemy's position
			Vector3 e2Position = Enemy2.transform.position;

			// Transform the position in our image format
			int e2X = TransformXCoordinate(e2Position.x);
			int e2Y = TransformYCoordinate(e2Position.y);

			// Add the enemy to the image
			FillSquareGivenPosition(e2Y, e2X, 0.25f);
		}
	}

	private void NormalizeData()
	{
		float minX = -24.25f;
		float maxX = 30.9f;

		float minY = -15.6f;
		float maxY = 12.5f;

		for(int i = 0; i < InputVector.GetLength(1); i+=2)
		{
			InputVector[0, i] = (InputVector[0, i] - minX) / (maxX - minX);
		}

		for(int i = 1; i < InputVector.GetLength(1); i+=2)
		{
			InputVector[0, i] = (InputVector[0, i] - minY) / (maxY - minY);
		}
	}

	private void NormalizeDataLSTM()
	{
		float minX = -24.25f;
		float maxX = 30.9f;

		float minY = -15.6f;
		float maxY = 12.5f;

		for(int i = 0; i < InputVectorLSTM.GetLength(2); i+=2)
		{
			InputVectorLSTM[0, CurrentFrameToGetData, i] = (InputVectorLSTM[0, CurrentFrameToGetData, i] - minX) / (maxX - minX);
		}

		for(int i = 1; i < InputVectorLSTM.GetLength(2); i+=2)
		{
			InputVectorLSTM[0, CurrentFrameToGetData, i] = (InputVectorLSTM[0, CurrentFrameToGetData, i] - minY) / (maxY - minY);
		}
	}

	private void RaiseAwareness()
	{
		for(int i = 0; i < InputVector.GetLength(1); i++)
		{
			if(InputVector[0, i] > 1f || InputVector[0, i] < 0f)
			{
				Debug.Log("NORMALIZING ERROR!!!!!");
				Debug.Log("Index: " + i);
				Debug.Log("Value: " + InputVector[0, i]);
			}
		}
	}

	private void RaiseAwarenessLSTM()
	{
		for(int i = 0; i < InputVectorLSTM.GetLength(2); i++)
		{
			if(InputVectorLSTM[0, CurrentFrameToGetData, i] > 1f || InputVectorLSTM[0, CurrentFrameToGetData, i] < 0f)
			{
				Debug.Log("NORMALIZING ERROR!!!!!");
				Debug.Log("Index: " + i);
				Debug.Log("Value: " + InputVectorLSTM[0, CurrentFrameToGetData, i]);
			}
		}
	}

	private float AngleToBall()
	{
		if(ball)
		{
			float vecX = ball.transform.position.x - gameObject.transform.position.x;
			float vecY = ball.transform.position.y - gameObject.transform.position.y;

			float answer = Mathf.Atan2(Mathf.Abs(vecY), Mathf.Abs(vecX));
			
			answer = answer * 180.0f / Mathf.PI;
			
			if(vecX > 0f && vecY < 0f)
				answer = 360.0f - answer;

			if(vecX < 0f && vecY > 0f)
				answer = 90.0f + answer;

			if(vecX < 0f && vecY < 0f)
				answer = answer + 180.0f;

			answer = answer / 360.0f;

			return answer;
		}
		return 0f;
	}

	private void CreateVectorFromData()
	{
		if(nNType == NNType.LSTM)
		{
			InputVectorLSTM[0, CurrentFrameToGetData, 0] = gameObject.transform.position.x;
			InputVectorLSTM[0, CurrentFrameToGetData, 1] = gameObject.transform.position.y;

			InputVectorLSTM[0, CurrentFrameToGetData, 2] = ball.transform.position.x;
			InputVectorLSTM[0, CurrentFrameToGetData, 3] = ball.transform.position.y;

			InputVectorLSTM[0, CurrentFrameToGetData, 4] = Player.transform.position.x;
			InputVectorLSTM[0, CurrentFrameToGetData, 5] = Player.transform.position.y;

			InputVectorLSTM[0, CurrentFrameToGetData, 6] = Enemy1.transform.position.x;
			InputVectorLSTM[0, CurrentFrameToGetData, 7] = Enemy1.transform.position.y;

			InputVectorLSTM[0, CurrentFrameToGetData, 8] = Enemy2.transform.position.x;
			InputVectorLSTM[0, CurrentFrameToGetData, 9] = Enemy2.transform.position.y;
		}
		if(nNType == NNType.VectorDataNN)
		{
			InputVector[0, 0] = gameObject.transform.position.x;
			InputVector[0, 1] = gameObject.transform.position.y;

			InputVector[0, 2] = ball.transform.position.x;
			InputVector[0, 3] = ball.transform.position.y;

			InputVector[0, 4] = Player.transform.position.x;
			InputVector[0, 5] = Player.transform.position.y;

			InputVector[0, 6] = Enemy1.transform.position.x;
			InputVector[0, 7] = Enemy1.transform.position.y;

			InputVector[0, 8] = Enemy2.transform.position.x;
			InputVector[0, 9] = Enemy2.transform.position.y;
		}


		// AddAngleAndDistanceData();
		if(nNType == NNType.VectorDataNN)
		{
			NormalizeData();
			RaiseAwareness();
		}
		if(nNType == NNType.LSTM)
		{
			NormalizeDataLSTM();
			RaiseAwarenessLSTM();
		}
	}

	private void AddAngleAndDistanceData()
	{
		float player_distance_x = gameObject.transform.position.x - ball.transform.position.x;
		float player_distance_y = gameObject.transform.position.y - ball.transform.position.y;

		float Enemy1_distance_x = Enemy1.transform.position.x - ball.transform.position.x;
		float Enemy1_distance_y = Enemy1.transform.position.y - ball.transform.position.y;

		float Enemy2_distance_x = Enemy2.transform.position.x - ball.transform.position.x;
		float Enemy2_distance_y = Enemy2.transform.position.y - ball.transform.position.y;

		float Companion_distance_x = Player.transform.position.x - ball.transform.position.x;
		float Companion_distance_y = Player.transform.position.y - ball.transform.position.y;

		InputVector[0, 10] = AngleToBall();

		InputVector[0, 11] = Mathf.Pow(player_distance_x, 2) + Mathf.Pow(player_distance_y, 2);
		InputVector[0, 12] = Mathf.Pow(Enemy1_distance_x, 2) + Mathf.Pow(Enemy1_distance_y, 2);
		InputVector[0, 13] = Mathf.Pow(Enemy2_distance_x, 2) + Mathf.Pow(Enemy2_distance_y, 2);
		InputVector[0, 14] = Mathf.Pow(Companion_distance_x, 2) + Mathf.Pow(Companion_distance_y, 2);
	}

	// Fills a square given a position in the terrain and the grey value (in grey scale) to fill it with
	void FillSquareGivenPosition(int row, int column, float grey)
	{
		// Fill the position of the player
		InputImage[0, row, column, 0] = grey;

		// Up and Down positions
		InputImage[0, row+1, column, 0] = grey;
		if(row > 0)
			InputImage[0, row-1, column, 0] = grey;
		
		// Left and right positions
		InputImage[0, row, column+1, 0] = grey;
		if(column > 0)
			InputImage[0, row, column-1, 0] = grey;

		// Corners
		InputImage[0, row+1, column+1, 0] = grey;
		if(column > 0)
			InputImage[0, row+1, column-1, 0] = grey;
		if(row > 0)
			InputImage[0, row-1, column+1, 0] = grey;
		if(row > 0 && column > 0)
			InputImage[0, row-1, column-1, 0] = grey;
	}

	// Load the CNN model
	void LoadModel()
	{
		// Import the graph, given in .bytes format
		graph.Import(ModelGraph.bytes);
		// Create a session with the graph
		session = new TFSession(graph);
	}

	private void OutputModel()
	{
		List<TFOperation> op_list = new List<TFOperation>( graph.GetEnumerator() );
		for(int i = 0; i < op_list.Count; i++)
		{
			Debug.Log(op_list[i].Name);
			Debug.Log(op_list[i].OpType);
		}
	}

	// Evaluate the model and infers a new input
	public void Evaluate()
	{
		// Output variable
		float[,] Output;

		// Obtain the runner of the session
		var runner = session.GetRunner();

		// Transform our Input array into a Tensorflow Tensor
		TFTensor Input;
		if(nNType == NNType.GrayScaleImageCNN)
		{
			Input = InputImage;
			runner.AddInput(graph["state"][0], Input);
		}
		if(nNType == NNType.VectorDataNN)
		{
			Input = InputVector;
			runner.AddInput(graph["state"][0], Input);
		}
		if(nNType == NNType.LSTM)
		{
			Input = InputVectorLSTM;
			runner.AddInput(graph["state_input"][0], Input);
		}

		// Set up the output tensor
		runner.Fetch(graph["action/Relu"][0]);

		// Run the model
		Output = runner.Run()[0].GetValue() as float[,];

		// Save the output
		TemporaryDirectionInGrades = Output[0,0];
	}

	// Getter function for the CNN infered value
	public float GetDirectionInGrades()
	{
		return DirectionInGrades;
	}
}
