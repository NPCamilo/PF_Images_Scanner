export interface PyodideStatus {
  status: 'idle' | 'loading-wasm' | 'loading-packages' | 'loading-dsp' | 'ready' | 'error';
  message: string;
}

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
};

/**
 * Cliente singleton que administra el Web Worker de Python/Pyodide sin bloquear la UI.
 */
class PyodideDspClient {
  private worker: Worker | null = null;
  private reqId = 0;
  private pending = new Map<number, PendingRequest>();
  private statusListeners = new Set<(status: PyodideStatus) => void>();
  private currentStatus: PyodideStatus = { status: 'idle', message: 'Motor Python sin iniciar' };

  public onStatusChange(listener: (status: PyodideStatus) => void): () => void {
    this.statusListeners.add(listener);
    listener(this.currentStatus);
    return () => this.statusListeners.delete(listener);
  }

  private ensureWorker(): Worker {
    if (typeof window === 'undefined') {
      throw new Error('PyodideDspClient solo puede ejecutarse en el navegador.');
    }
    if (!this.worker) {
      this.worker = new Worker('/workers/pyodide-worker.js');
      this.worker.onmessage = (event: MessageEvent) => {
        const data = event.data as {
          type?: string;
          status?: PyodideStatus['status'];
          message?: string;
          id?: number;
          ok?: boolean;
          result?: unknown;
          error?: string;
        };

        if (data.type === 'status' && data.status) {
          this.currentStatus = { status: data.status, message: data.message ?? '' };
          this.statusListeners.forEach((fn) => fn(this.currentStatus));
          return;
        }

        if (typeof data.id === 'number') {
          const entry = this.pending.get(data.id);
          if (!entry) return;
          this.pending.delete(data.id);
          if (data.ok) {
            entry.resolve(data.result);
          } else {
            entry.reject(new Error(data.error ?? 'Error desconocido en Pyodide Worker'));
          }
        }
      };
    }
    return this.worker;
  }

  public call<T = unknown>(action: string, payload: unknown = {}): Promise<T> {
    const worker = this.ensureWorker();
    const id = ++this.reqId;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      worker.postMessage({ id, action, payload });
    });
  }
}

export const pyodideDsp = new PyodideDspClient();
