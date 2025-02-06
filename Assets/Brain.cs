using UnityEngine;
using System.Linq;
using System.Collections.Generic;
using Unity.VisualScripting;
using TMPro;
using UnityEngine.UIElements;


public class Replay
{
    public List<double> states;
    public double reward;

    public Replay(double xr, double ballz, double ballvx, double r)
    {
        states = new() { xr, ballz, ballvx };
        reward = r;
    }
}


public class Brain : MonoBehaviour
{
    // Canvas text elements.
    [SerializeField] TMP_Text statsText;
    [SerializeField] TMP_Text failsText;
    [SerializeField] TMP_Text decayText;
    [SerializeField] TMP_Text bestText;
    [SerializeField] TMP_Text thisText;

    [SerializeField] GameObject ball;   // Object to monitor.

    ANN ann;                            // 
    [SerializeField] float timeScale = 5;

    float reward = 0;                   // Reward to associate with actions.
    List<Replay> replayMemory = new();  // Memory - list of past actions and rewards.
    int mCapacity = 10000;              // Memory capacity.

    float discount = 0.99f;             // How much future states affect rewards.
    [SerializeField] float exploreRate = 100;            // Chance of picking random action.
    float maxExploreRate = 100;         // Max chance value.
    float minExploreRate = 0.01f;       // Min chance value.
    [SerializeField] float exploreDecay = 0.0001f;       // Chance decay amount for each update.

    Vector3 ballStartPos;               // Record start position of object.
    int failCount = 0;                  // Count when the ball is dropped.
    float tiltSpeed = 0.5f;             // Max angle to apply to tilting each update:
                                        // Make sure this is large enough so that the q value
                                        // multiplied by it is enough to recover balance
                                        // when the ball gets a good speed up.
    float timer = 0;                    // Timer to keep track of balancing
    float maxBalanceTime = 0;           // Record time ball is kept balanced


    void Start()
    {
        ann = new(3, 2, 1, 6, 0.2f);
        ballStartPos = ball.transform.position;
        Time.timeScale = timeScale;
        UpdateTexts();
    }

    void Update()
    {
        if (Input.GetKeyDown("space"))
        {
            ResetBall();
        }
    }

    void FixedUpdate()
    {
        timer += Time.deltaTime;
        List<double> states = new();
        List<double> qs = new();

        states.Add(this.transform.rotation.x);
        states.Add(ball.transform.position.z);
        states.Add(ball.GetComponent<Rigidbody>().angularVelocity.x);

        qs = SoftMax(ann.CalcOutput(states));
        double maxQ = qs.Max();
        int maxQIndex = qs.ToList().IndexOf(maxQ);
        exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);

        if (Random.Range(0, 100) < exploreRate)
        {
            maxQIndex = Random.Range(0, 2);
        }

        if (maxQIndex == 0)
        {
            transform.Rotate(Vector3.right, tiltSpeed * (float)qs[maxQIndex]);
        }
        else if (maxQIndex == 1)
        {
            transform.Rotate(Vector3.right, -tiltSpeed * (float)qs[maxQIndex]);
        }

        if (ball.GetComponent<BallState>().dropped)
        { reward = -1.0f; }
        else
        { reward = 0.1f; }

        Replay lastMemory = new(this.transform.rotation.x,
                                ball.transform.position.z,
                                ball.GetComponent<Rigidbody>().angularVelocity.x,
                                reward);

        if (replayMemory.Count > mCapacity)
        {
            replayMemory.RemoveAt(0);
        }

        replayMemory.Add(lastMemory);

        if (ball.GetComponent<BallState>().dropped)
        {
            for (int i = replayMemory.Count - 1; i >= 0; i--)
            {
                List<double> toutputsOld = new();
                List<double> toutputsNew = new();
                toutputsOld = SoftMax(ann.CalcOutput(replayMemory[i].states));

                double maxQOld = toutputsOld.Max();
                int action = toutputsOld.ToList().IndexOf(maxQOld);

                double feedback;
                if (i == replayMemory.Count - 1 || replayMemory[i].reward == -1)
                {
                    feedback = replayMemory[i].reward;
                }
                else
                {
                    toutputsNew = SoftMax(ann.CalcOutput(replayMemory[i + 1].states));
                    maxQ = toutputsNew.Max();
                    feedback = replayMemory[i].reward + discount * maxQ;
                }

                toutputsOld[action] = feedback;
                ann.Train(replayMemory[i].states, toutputsOld);
            }

            ball.GetComponent<BallState>().dropped = false;
            transform.rotation = Quaternion.identity;
            ResetBall();
            replayMemory.Clear();
            failCount++;
        }

        UpdateTexts();
    }

    List<double> SoftMax(List<double> values)
    {
        double max = values.Max();

        float scale = 0;
        for (int i = 0; i < values.Count; i++)
        {
            scale += Mathf.Exp((float)(values[i] - max));
        }

        List<double> result = new();
        for (int i = 0; i < values.Count; i++)
        {
            result.Add(Mathf.Exp((float)(values[i] - max)) / scale);
        }

        return result;
    }

    void ResetBall()
    {
        ball.transform.position = ballStartPos;
        ball.GetComponent<Rigidbody>().linearVelocity = new(0, 0, 0);
        ball.GetComponent<Rigidbody>().angularVelocity = new(0, 0, 0);

        if (timer > maxBalanceTime)
        {
            maxBalanceTime = timer;
        }

        timer = 0;
    }

    void UpdateTexts()
    {
        statsText.SetText("-- STATS --");
        failsText.SetText("Fails: " + failCount);
        decayText.SetText("Decay Rate: " + exploreRate);
        bestText.SetText("Best Balance: " + maxBalanceTime);
        thisText.SetText("This Balance: " + timer);
    }
}
