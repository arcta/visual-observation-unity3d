using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleEnvironmentModel : Challenge
{
    //public List<GameObject> prefabList = new List<GameObject>();
    //public List<Material> materiaList = new List<Material>();
    public AgentCamera agent;


    void Start()
    {
        // position the camera
        agent.transform.position = new Vector3(0, 1f, 0);
        agent.transform.rotation = Quaternion.identity;
        agent.GetView().transform.rotation = Quaternion.identity;
    }

    public override void Reset()
    {
        Start();
    }

    public override float GetReward(float[] vectorAction, string textAction)
    {
        // no reward
        return 0f;
    }
}
