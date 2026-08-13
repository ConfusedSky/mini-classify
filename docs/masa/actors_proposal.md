# Actor Proposal

## Proposal

Right now there is a lot of things that can be done at the same time that we
currently aren't doing. I propose that we build out a series of 9 actors that
each handle a stage of the process. This also should help simplify things
because right now there is a huge mess of intertwined processes and
conditionals, without many **hard** boundaries to break things up.

## Design

Each of these actors should have their own message queues and be implemented in
their own files. These new files should live under src/ instead of just being
loose files. Each actor should also be it's own thread, it's possible it might
need to even be it's own process, dependent on followup spikes.

### Walker

* This actor looks for the walk cache if it doesn't exist or `--rescan` is
  called.
  * If the cache is found it sends all the files down through to `Cache
    Checker's` queue.
  * If the cache is not found it walks the directories and sends files one at a
    time through to `Cache Checker's` queue.

### Cache Checker

* When a file comes in through the queue:
  * It first checks the embedding cache, if the embedding exists and needs render
    is false, then the file is dropped
  * Then it checks the pose cache, if the pose is cached then pass the file with
    a `"kind": "embed"` with the pose metadata to the `Loader's` queue.
  * Finally if neither of the above then the file with a `"kind": "pose"` is
    passed to the loaders queue.

### Loader

* Uses up to [loader_worker_count=4] loader workers to load the mesh into the gpu
* Passes along kind and pose metadata to the `Renderer`
* Loads up to [loader_preload_cache=2Gib] of meshes onto the gpu
* Holds on to the mesh in memory until the `Renderer` releases the memory.

### Renderer

* Receives a loaded mesh plus metadata signifying what kind of render this is
* If the `"kind": "pose"` then each image required to handing posing is rendered
* If the `"kind": "embed"` then each of the images according to the pose data
  and the input parameters are generated.
* At this point the if `--render-dir` is specified the renders are saved to disk
* Pose renders are then sent to `Poser` and embed renders are sent to `Embedder`.

### Poser

* Handles posing the way it currently works, if needs arbitration the results
  are sent to `Arbitrator`
* Request embedding from the `Embedder` to handle the ensemble.
* Once posing is complete send back to `Renderer` with `"kind": "embed"`.

### Arbitrator

* Handles queuing up async requests to the arbitrator, appropriately batches and
  times requests to not be rate limited
* Once the arbitration is complete the result is passed back to the `Poser`.

### Embedder

* Embeds an image then passes it either back to `Poser` to complete the ensemble
  or sends it to `Done`

### Done

* Handles everything that happens after embedding (saving to cache, etc)
