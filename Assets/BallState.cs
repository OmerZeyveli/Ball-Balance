using UnityEngine;

public class BallState : MonoBehaviour
{
    public bool dropped = false;

    void OnCollisionEnter(Collision other) {
        if(other.gameObject.CompareTag("drop"))
        {
            dropped = true;
        }
    }
}
