using UnityEngine;

public class BallState : MonoBehaviour
{
    public bool dropped = false; // Has the ball dropped?

    void OnCollisionEnter(Collision other) {
        if(other.gameObject.CompareTag("drop")) // Hit "drop" object?
        {
            dropped = true; // Mark as dropped
        }
    }
}
