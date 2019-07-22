using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public abstract class Challenge : MonoBehaviour
{
    public float maxReward;

    void Start()
    {
        Reset();
    }

    void Update()
    {
        FixedUpdate();
    }

    public virtual float GetReward(float[] vectorAction, string textAction)
    {
        return 0f;
    }

    public virtual void Reset() { Debug.Log("Challenge-Reset"); }

    public virtual void FixedUpdate() { }
}
