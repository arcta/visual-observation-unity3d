using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class AgentCamera : Agent
{
    public float turnSpeed = 1f;
    public Camera view;
    public Challenge challenge;


    public override void AgentReset()
    {
        // no reset here: delegate to the active challenge
        challenge.Reset();
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        // make the move
        Azimuth(vectorAction[0]);
        Altitude(vectorAction[1]);
        // get rewarded
        AddReward(challenge.GetReward(vectorAction, textAction));
    }

    public Camera GetView()
    {
        return view;
    }

    public Vector3 GetFocus()
    {
        return view.transform.forward;
    }

    public Ray GetViewPoint(float rX, float rY)
    {
        return view.ViewportPointToRay(new Vector3(rX, rY, 0));
    }

    public RayPerception GetPerception()
    {
        return GetComponent<RayPerception>();
    }

    void Azimuth(float value)
    {
        // horizontal turn
        transform.Rotate(0, turnSpeed * value, 0);
    }

    void Altitude(float value)
    {
        // frontside up and down tilt (no flip over allowed)
        float tilt = view.transform.localRotation.eulerAngles.x;
        // limit tilt down
        if (tilt > 80f && tilt < 90f && value > 0f)
            value *= Random.Range(-1f, -10f);
        // limit tilt up
        else if (tilt < 280f && tilt > 270f && value < 0f)
            value *= Random.Range(-1f, -10f);
        view.transform.Rotate(turnSpeed * value, 0, 0);
    }
}
