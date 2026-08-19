"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { checkBackendHealth, BackendHealthResponse } from "../lib/api";

export type BackendConnectionStatus = "checking" | "connected" | "disconnected";

export interface UseBackendStatusReturn {
  status: BackendConnectionStatus;
  isWakingUp: boolean;
  isConnected: boolean;
  isDisconnected: boolean;
  elapsedSeconds: number;
  latencyMs: number | null;
  retryCount: number;
  errorMessage: string | null;
  healthData: BackendHealthResponse | null;
  retryConnection: () => Promise<void>;
}

export function useBackendStatus(): UseBackendStatusReturn {
  const [status, setStatus] = useState<BackendConnectionStatus>("checking");
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [retryCount, setRetryCount] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [healthData, setHealthData] = useState<BackendHealthResponse | null>(null);

  const isPollingRef = useRef<boolean>(false);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const elapsedIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const keepAliveIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const consecutiveFailuresRef = useRef<number>(0);
  const startTimeRef = useRef<number>(Date.now());

  const cleanupTimers = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    if (elapsedIntervalRef.current) {
      clearInterval(elapsedIntervalRef.current);
      elapsedIntervalRef.current = null;
    }
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current);
      keepAliveIntervalRef.current = null;
    }
  };

  const startElapsedTimer = () => {
    if (elapsedIntervalRef.current) return;
    startTimeRef.current = Date.now();
    elapsedIntervalRef.current = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
  };

  const stopElapsedTimer = () => {
    if (elapsedIntervalRef.current) {
      clearInterval(elapsedIntervalRef.current);
      elapsedIntervalRef.current = null;
    }
  };

  const performCheck = useCallback(async (isInitial = false): Promise<boolean> => {
    try {
      const res = await checkBackendHealth(7000);
      if (res.ok) {
        consecutiveFailuresRef.current = 0;
        setHealthData(res);
        setLatencyMs(res.latencyMs || null);
        setStatus("connected");
        setErrorMessage(null);
        stopElapsedTimer();
        return true;
      } else {
        consecutiveFailuresRef.current += 1;
        setErrorMessage(res.error || "Backend is offline or waking up");
        return false;
      }
    } catch (e: any) {
      consecutiveFailuresRef.current += 1;
      setErrorMessage(e.message || "Failed to reach backend");
      return false;
    }
  }, []);

  const startPolling = useCallback(() => {
    if (isPollingRef.current) return;
    isPollingRef.current = true;
    setStatus("checking");
    startElapsedTimer();

    const poll = async () => {
      setRetryCount((prev) => prev + 1);
      const isOk = await performCheck();
      if (isOk) {
        isPollingRef.current = false;
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        return;
      }

      // Check timeout (after 120s of continuous wake-up attempts)
      const currentElapsed = (Date.now() - startTimeRef.current) / 1000;
      if (currentElapsed > 120) {
        isPollingRef.current = false;
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        stopElapsedTimer();
        setStatus("disconnected");
        setErrorMessage("Server wake-up timed out after 2 minutes. Please check backend logs or retry.");
      }
    };

    pollIntervalRef.current = setInterval(poll, 2500);
  }, [performCheck]);

  const retryConnection = useCallback(async () => {
    cleanupTimers();
    isPollingRef.current = false;
    consecutiveFailuresRef.current = 0;
    setRetryCount(0);
    setElapsedSeconds(0);
    setStatus("checking");
    setErrorMessage(null);

    const isOk = await performCheck();
    if (!isOk) {
      startPolling();
    }
  }, [performCheck, startPolling]);

  // Keep-alive loop: once connected, send lightweight ping every 40s to prevent spin-down
  useEffect(() => {
    if (status === "connected") {
      keepAliveIntervalRef.current = setInterval(async () => {
        const isOk = await performCheck();
        if (!isOk && consecutiveFailuresRef.current >= 2) {
          // Backend might have spun down again or dropped
          setStatus("checking");
          startPolling();
        }
      }, 40000);
    }
    return () => {
      if (keepAliveIntervalRef.current) {
        clearInterval(keepAliveIntervalRef.current);
        keepAliveIntervalRef.current = null;
      }
    };
  }, [status, performCheck, startPolling]);

  // Initial check on mount
  useEffect(() => {
    let isMounted = true;
    (async () => {
      setStatus("checking");
      startElapsedTimer();
      const isOk = await performCheck(true);
      if (isMounted) {
        if (!isOk) {
          startPolling();
        }
      }
    })();

    return () => {
      isMounted = false;
      cleanupTimers();
    };
  }, [performCheck, startPolling]);

  return {
    status,
    isWakingUp: status === "checking",
    isConnected: status === "connected",
    isDisconnected: status === "disconnected",
    elapsedSeconds,
    latencyMs,
    retryCount,
    errorMessage,
    healthData,
    retryConnection
  };
}
