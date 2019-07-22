using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleVisualDetection : Challenge
{
    public List<GameObject> prefabList = new List<GameObject>();
    public List<Material> materiaList = new List<Material>();
    public AgentCamera agent;
    GameObject target;


    void Start()
    {
        maxReward = 10f;
        // position the camera
        agent.transform.position = new Vector3(0, 1f, 0);
    }

    float GetRandomCoord(float minDistance, float maxDistance)
    {
        float d = Random.Range(minDistance, maxDistance);
        if (Random.Range(0f, 1f) < 0.5f)
            return -d;
        return d;
    }

    void GetRandomObject()
    {
        int prefabIndex = Random.Range(0, prefabList.Count);
        int materialIndex = Random.Range(0, materiaList.Count);
        float x = GetRandomCoord(1.5f, 10f);
        float z = GetRandomCoord(1.5f, 10f);
        target = Instantiate(prefabList[prefabIndex], new Vector3(x, 1.5f, z), Random.rotation) as GameObject;
        // randomize material
        target.GetComponent<Renderer>().material = materiaList[materialIndex];
        // randomize shape
        target.transform.localScale = new Vector3(Random.Range(0.5f, 2.5f), Random.Range(0.5f, 2.5f), Random.Range(0.5f, 2.5f));
    }

    public override void Reset()
    {
        // no camera reset here: start from wherever it stopped in the previous episode
        Destroy(target);
        GetRandomObject();
    }

    public override float GetReward(float[] vectorAction, string textAction)
    {
        // get agent view and direction to the target
        Vector3 toTarget = target.transform.position - agent.transform.position;
        float rightDirection = Vector3.Dot(agent.GetFocus(), toTarget) / toTarget.magnitude;
        if (rightDirection > 0.3f) // object is within the view
        {
            // agent focused on the target
            if (rightDirection > 0.95f)
                return maxReward;
            // penalize turn away from the visible target
            float rightTurn = Vector3.Dot(agent.GetView().transform.right, toTarget) * vectorAction[0];
            float rightTilt = Vector3.Dot(agent.GetView().transform.up, toTarget) * vectorAction[1];
            float reward = rightDirection * rightTurn * rightTilt;
            if (rightTurn < 0 && rightTilt < 0)
                return -2f * reward;
            return reward;
        }
        // if target is out off the view: keep soft preference for the horizon direction
        return -Mathf.Pow(Vector3.Dot(agent.GetFocus(), agent.transform.up), 2);
    }
}
